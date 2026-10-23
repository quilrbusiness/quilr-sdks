r"""Quilr Guardrails Integration for LiteLLM.

This module provides a custom guardrail class that integrates Quilr's guardrails API
with LiteLLM proxy for both request-side and response-side content checking.

Environment Variables:
    QUILR_GUARDRAILS_KEY: API key for Quilr guardrails (required)
    QUILR_GUARDRAILS_BASE_URL: Base URL for Quilr guardrails API
        (default: https://guardrails.quilr.ai)
    QUILR_GUARDRAILS_TIMEOUT: Timeout in seconds for guardrails API calls
        (default: 3). If timeout or any API error occurs, request passes through.
    APPLY_QUILR_GUARDRAILS_FOR_MODELS: Comma-separated list of models to apply
        guardrails to (optional, if not set applies to all)
    APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES: Comma-separated list of API key names
        to apply guardrails to (optional, if not set applies to all)

Usage in LiteLLM config.yaml:

    guardrails:
      - guardrail_name: "quilr-input"
        litellm_params:
          guardrail: quilr_litellm_guardrails.QuilrGuardrail
          mode: "pre_call"              # checks input before LLM call

      - guardrail_name: "quilr-input-duringcall"
        litellm_params:
          guardrail: quilr_litellm_guardrails.QuilrGuardrail
          mode: "during_call"           # checks input in parallel with LLM call

      - guardrail_name: "quilr-output"
        litellm_params:
          guardrail: quilr_litellm_guardrails.QuilrGuardrail
          mode: "post_call"             # checks output after LLM call

Usage in API requests:

    curl -X POST http://localhost:4000/v1/chat/completions \
      -H "Authorization: Bearer sk-xxx" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "guardrails": ["quilr-input", "quilr-output"]
      }'

Behavior:
    - quilr-input (pre_call): Checks messages before sending to LLM
        - blocked: Request rejected with error
        - redacted: Messages replaced with redacted version
        - safe: Passes through unchanged

    - quilr-input-duringcall (during_call): Checks messages in parallel with LLM call
        - Runs concurrently with LLM call (no added latency)
        - Response not returned until both guardrail and LLM complete
        - blocked: Response rejected with error
        - redacted: Messages replaced with redacted version
        - safe: Passes through unchanged

    - quilr-output (post_call): Checks LLM response before returning to user
        - blocked: Response rejected with error
        - redacted: Response content replaced with redacted version
        - safe: Passes through unchanged

All API formats (QuilrUnifiedGuardrail):
    QuilrGuardrail reads only the OpenAI Chat Completions and Responses
    formats. To also check other formats (e.g. Anthropic /v1/messages,
    Gemini generateContent), use QuilrUnifiedGuardrail in the same config
    instead. It uses LiteLLM's apply_guardrail interface and needs
    a LiteLLM version that supports it (tested with 1.106.2):

          guardrail: quilr_litellm_guardrails.QuilrUnifiedGuardrail
"""

import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import RejectedRequestError
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.llms.openai import ResponsesAPIResponse

# Used only by QuilrUnifiedGuardrail. Older LiteLLM versions don't have this
# exception (or apply_guardrail support); the fallback keeps this module
# importable there, so QuilrGuardrail keeps working.
try:
    from litellm.exceptions import GuardrailRaisedException
except ImportError:
    GuardrailRaisedException = RejectedRequestError


class QuilrGuardrail(CustomGuardrail):
    """Custom guardrail that integrates with Quilr's guardrails API.

    Handles both request-side (input) and response-side (output) checking:
    - Request-side: Checks messages before sending to LLM, can redact or block
    - Response-side: Checks LLM output before returning to user, can redact or block
    """

    def __init__(self, **kwargs):
        """Initialize the Quilr guardrail with configuration from environment variables."""
        self.api_key = os.getenv("QUILR_GUARDRAILS_KEY")
        self.api_base = os.getenv(
            "QUILR_GUARDRAILS_BASE_URL", "https://guardrails.quilr.ai"
        )
        self.timeout = int(os.getenv("QUILR_GUARDRAILS_TIMEOUT", "3"))

        # Parse optional filters (comma-separated lists)
        models_env = os.getenv("APPLY_QUILR_GUARDRAILS_FOR_MODELS", "")
        self.allowed_models = [
            m.strip() for m in models_env.split(",") if m.strip()
        ] or None

        key_names_env = os.getenv("APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES", "")
        self.allowed_key_names = [
            k.strip() for k in key_names_env.split(",") if k.strip()
        ] or None

        if not self.api_key:
            verbose_proxy_logger.warning(
                "QUILR_GUARDRAILS_KEY not set. Quilr guardrail will not function."
            )

        super().__init__(**kwargs)

    def _serialize_user_api_key_dict(
        self, user_api_key_dict: UserAPIKeyAuth
    ) -> Dict[str, Any]:
        """Serialize UserAPIKeyAuth, excluding None values and non-serializable fields.

        Args:
            user_api_key_dict: API key authentication info

        Returns:
            Dictionary with serializable fields only

        """
        dumped = user_api_key_dict.model_dump(exclude_none=True)
        result = {}
        for key, value in dumped.items():
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                pass  # Skip non-serializable fields
        return result

    async def _call_quilr_api(
        self, payload: Dict[str, Any], user_api_key_dict: UserAPIKeyAuth
    ) -> Dict[str, Any]:
        """Make a request to Quilr's guardrails API.

        Args:
            payload: Request body (either {"messages": [...]} or {"text": "..."})
            user_api_key_dict: API key authentication info to include as metadata

        Returns:
            API response as dictionary

        """
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        endpoint = f"{self.api_base.rstrip('/')}/sdk/v1/check"

        # Add metadata with caller info and user API key details
        payload["metadata"] = {
            "caller": "litellm",
            **self._serialize_user_api_key_dict(user_api_key_dict),
        }

        response = await async_client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    def _format_block_message(self, categories: List[str]) -> str:
        """Format a block message including detected categories."""
        if categories:
            return f"Content blocked by Quilr: {', '.join(categories)} detected"
        return "Content blocked by Quilr"

    def _responses_input_to_messages(
        self, input_data: Union[str, List[Dict]], instructions: Optional[str]
    ) -> List[Dict[str, str]]:
        """Convert OpenAI Responses API input format to chat messages format.

        Args:
            input_data: Either a string or array of message objects
            instructions: Optional system instructions

        Returns:
            List of message dictionaries in chat format

        """
        messages = []

        # Add instructions as system message if present
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Handle input
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            messages.extend(input_data)

        return messages

    def _messages_to_responses_input(
        self, messages: List[Dict[str, str]], original_input_was_string: bool
    ) -> tuple[Optional[str], Union[str, List[Dict], None]]:
        """Convert chat messages format back to OpenAI Responses API format.

        Args:
            messages: List of message dictionaries in chat format
            original_input_was_string: Whether the original input was a string

        Returns:
            Tuple of (instructions, input) in Responses API format

        """
        instructions = None
        input_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                instructions = msg.get("content")
            else:
                input_messages.append(msg)

        # If original input was a string and we have exactly one user message,
        # return it as a string
        if original_input_was_string and len(input_messages) == 1:
            return instructions, input_messages[0].get("content")

        return instructions, input_messages if input_messages else None

    def _should_apply_guardrail(
        self, model: Optional[str], user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        """Check if guardrail should be applied based on filters.

        Returns True if:
        - No filters are configured (apply to all), OR
        - Request matches ALL configured filters (AND logic)

        Args:
            model: The model being called
            user_api_key_dict: API key authentication info

        Returns:
            True if guardrail should be applied, False to skip

        """
        # If no filters configured, apply to all
        if not self.allowed_models and not self.allowed_key_names:
            return True

        # Check model filter
        if self.allowed_models:
            if not model or model not in self.allowed_models:
                return False

        # Check key name filter
        if self.allowed_key_names:
            key_name = user_api_key_dict.key_name
            if not key_name or key_name not in self.allowed_key_names:
                return False

        return True

    def _extract_messages_from_request(
        self, data: dict, is_responses_api: bool
    ) -> tuple[Optional[List[Dict]], bool]:
        """Extract messages from request data.

        Args:
            data: Request data dictionary
            is_responses_api: Whether this is a Responses API call

        Returns:
            Tuple of (messages, original_input_was_string). Returns (None, False) if no messages.

        """
        if is_responses_api:
            input_data = data.get("input")
            instructions = data.get("instructions")
            if not input_data:
                return None, False
            original_input_was_string = isinstance(input_data, str)
            messages = self._responses_input_to_messages(input_data, instructions)
            return messages, original_input_was_string

        messages = data.get("messages")
        return messages if messages else None, False

    def _apply_redacted_messages(
        self,
        data: dict,
        redacted_messages: List[Dict],
        is_responses_api: bool,
        original_input_was_string: bool,
    ) -> None:
        """Apply redacted messages back to request data.

        Args:
            data: Request data dictionary to modify
            redacted_messages: Redacted messages from guardrail API
            is_responses_api: Whether this is a Responses API call
            original_input_was_string: Whether the original input was a string

        """
        if is_responses_api:
            new_instructions, new_input = self._messages_to_responses_input(
                redacted_messages, original_input_was_string
            )
            if new_instructions is not None:
                data["instructions"] = new_instructions
            elif "instructions" in data:
                del data["instructions"]
            if new_input is not None:
                data["input"] = new_input
        else:
            data["messages"] = redacted_messages

    def _collect_output_text_items(self, response: ResponsesAPIResponse) -> List:
        """Collect all output_text items from a Responses API response.

        Args:
            response: ResponsesAPIResponse object

        Returns:
            List of output_text content items

        """
        output_text_items = []
        if not response.output:
            return output_text_items

        for output_item in response.output:
            if not hasattr(output_item, "content") or not output_item.content:
                continue
            for content_item in output_item.content:
                if (
                    not hasattr(content_item, "type")
                    or content_item.type != "output_text"
                ):
                    continue
                if hasattr(content_item, "text") and content_item.text:
                    output_text_items.append(content_item)

        return output_text_items

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "responses",
            "aresponses",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """Check request content before sending to LLM.

        Supports both Chat Completions API (messages) and Responses API (input/instructions).

        - If blocked: raises exception
        - If redacted: replaces messages/input with redacted version
        - If safe: passes through unchanged
        """
        verbose_proxy_logger.info("Quilr pre-call: request received")

        if not self.api_key:
            return data

        if not self._should_apply_guardrail(data.get("model"), user_api_key_dict):
            verbose_proxy_logger.info("Quilr pre-call: skipped (filter)")
            return data

        is_responses_api = call_type in ("responses", "aresponses")
        messages, original_input_was_string = self._extract_messages_from_request(
            data, is_responses_api
        )
        if not messages:
            return data

        try:
            result = await self._call_quilr_api(
                {"messages": messages}, user_api_key_dict
            )
            status = result.get("status", "safe")
            categories = result.get("categories_detected", [])

            if status == "blocked":
                verbose_proxy_logger.info("Quilr pre-call: blocked")
                raise RejectedRequestError(
                    message=self._format_block_message(categories),
                    model=data.get("model", ""),
                    llm_provider="quilr_guardrail",
                    request_data=data,
                )

            if status == "redacted":
                redacted_messages = result.get("messages")
                if redacted_messages:
                    self._apply_redacted_messages(
                        data,
                        redacted_messages,
                        is_responses_api,
                        original_input_was_string,
                    )
                    verbose_proxy_logger.info("Quilr pre-call: redacted")

            verbose_proxy_logger.info("Quilr pre-call: passed")
            return data

        except RejectedRequestError:
            raise
        except Exception:
            return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "responses",
            "aresponses",
        ],
    ):
        """Check request content in parallel with LLM call (during_call mode).

        Supports both Chat Completions API (messages) and Responses API (input/instructions).

        Runs concurrently with the LLM call. Response is not returned until
        both the guardrail check and LLM call complete.

        - If blocked: raises exception (response discarded)
        - If redacted: replaces messages/input with redacted version
        - If safe: passes through unchanged
        """
        verbose_proxy_logger.info("Quilr during-call: request received")

        if not self.api_key:
            return

        if not self._should_apply_guardrail(data.get("model"), user_api_key_dict):
            verbose_proxy_logger.info("Quilr during-call: skipped (filter)")
            return

        is_responses_api = call_type in ("responses", "aresponses")
        messages, original_input_was_string = self._extract_messages_from_request(
            data, is_responses_api
        )
        if not messages:
            return

        try:
            result = await self._call_quilr_api(
                {"messages": messages}, user_api_key_dict
            )
            status = result.get("status", "safe")
            categories = result.get("categories_detected", [])

            if status == "blocked":
                verbose_proxy_logger.info("Quilr during-call: blocked")
                raise RejectedRequestError(
                    message=self._format_block_message(categories),
                    model=data.get("model", ""),
                    llm_provider="quilr_guardrail",
                    request_data=data,
                )

            if status == "redacted":
                redacted_messages = result.get("messages")
                if redacted_messages:
                    self._apply_redacted_messages(
                        data,
                        redacted_messages,
                        is_responses_api,
                        original_input_was_string,
                    )
                    verbose_proxy_logger.info("Quilr during-call: redacted")

            verbose_proxy_logger.info("Quilr during-call: passed")

        except RejectedRequestError:
            raise
        except Exception:
            return

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """Check LLM response before returning to user.

        Supports both Chat Completions API (ModelResponse) and Responses API (ResponsesAPIResponse).

        - If blocked: raises exception
        - If redacted: modifies response content with redacted version
        - If safe: passes through unchanged
        """
        verbose_proxy_logger.info("Quilr post-call: response received")

        if not self.api_key:
            return response

        # Check if guardrail should be applied based on filters
        if not self._should_apply_guardrail(data.get("model"), user_api_key_dict):
            verbose_proxy_logger.info("Quilr post-call: skipped (filter)")
            return response

        # Handle OpenAI Responses API format
        if isinstance(response, ResponsesAPIResponse):
            return await self._check_responses_api_output(
                response, data, user_api_key_dict
            )

        # Handle Chat Completions API format
        if isinstance(response, litellm.ModelResponse):
            return await self._check_chat_completions_output(
                response, data, user_api_key_dict
            )

        return response

    async def _check_chat_completions_output(
        self,
        response: litellm.ModelResponse,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> litellm.ModelResponse:
        """Check output for Chat Completions API responses."""
        for choice in response.choices:
            if not hasattr(choice, "message") or not choice.message:
                continue

            content = choice.message.content
            if not content or not isinstance(content, str):
                continue

            try:
                result = await self._call_quilr_api(
                    {"text": content}, user_api_key_dict
                )
                status = result.get("status", "safe")
                categories = result.get("categories_detected", [])

                if status == "blocked":
                    verbose_proxy_logger.info("Quilr post-call: blocked")
                    raise RejectedRequestError(
                        message=self._format_block_message(categories),
                        model=data.get("model", ""),
                        llm_provider="quilr_guardrail",
                        request_data=data,
                    )

                if status == "redacted":
                    processed_text = result.get("processed_text")
                    if processed_text:
                        choice.message.content = processed_text
                        verbose_proxy_logger.info("Quilr post-call: redacted")

            except RejectedRequestError:
                raise
            except Exception:
                pass

        verbose_proxy_logger.info("Quilr post-call: passed")
        return response

    async def _check_responses_api_output(
        self,
        response: ResponsesAPIResponse,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> ResponsesAPIResponse:
        """Check output for OpenAI Responses API responses."""
        output_text_items = self._collect_output_text_items(response)

        if not output_text_items:
            verbose_proxy_logger.info("Quilr post-call: no output text found")
            return response

        all_text = "\n".join(item.text for item in output_text_items)

        try:
            result = await self._call_quilr_api({"text": all_text}, user_api_key_dict)
            status = result.get("status", "safe")
            categories = result.get("categories_detected", [])

            if status == "blocked":
                verbose_proxy_logger.info("Quilr post-call: blocked")
                raise RejectedRequestError(
                    message=self._format_block_message(categories),
                    model=data.get("model", ""),
                    llm_provider="quilr_guardrail",
                    request_data=data,
                )

            if status == "redacted":
                processed_text = result.get("processed_text")
                if processed_text:
                    output_text_items[0].text = processed_text
                    for item in output_text_items[1:]:
                        item.text = ""
                    verbose_proxy_logger.info("Quilr post-call: redacted")

        except RejectedRequestError:
            raise
        except Exception:
            return response

        verbose_proxy_logger.info("Quilr post-call: passed")
        return response


class QuilrUnifiedGuardrail(QuilrGuardrail):
    """Quilr guardrail built on LiteLLM's unified ``apply_guardrail`` interface.

    Use this class to check API formats that ``QuilrGuardrail`` does not read,
    such as Anthropic ``/v1/messages`` and Gemini ``generateContent``.
    LiteLLM extracts the texts from each format it has a guardrail translator
    for, passes them to ``apply_guardrail``, and writes the checked texts back.

    Behavior (same Quilr endpoint and results as ``QuilrGuardrail``):
        - Request: user texts are checked. If LiteLLM gives no message roles
          (e.g. Gemini), all texts are checked.
        - Response: each text is checked.
        - blocked: the call is rejected with an error
        - redacted: texts are replaced with Quilr's redacted versions
        - safe: texts pass through unchanged
        - Quilr call fails (timeout, connection or API error): texts pass
          through unchanged

    Requires a LiteLLM version with ``apply_guardrail`` support
    (tested with 1.106.2).
    """

    async def apply_guardrail(
        self,
        inputs: Dict[str, Any],
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Check the texts LiteLLM extracted from a request or response.

        Called by LiteLLM's guardrail translator for the request's API format.

        Args:
            inputs: Extracted content. Uses ``texts`` (list of strings) and,
                for requests, ``structured_messages`` (messages with roles)
            request_data: The original request data (model, metadata, ...)
            input_type: "request" for the prompt, "response" for the answer
            logging_obj: LiteLLM logging object (unused)

        Returns:
            ``inputs`` with ``texts`` replaced by the checked texts. LiteLLM
            requires the list to keep the same length and order.

        Raises:
            GuardrailRaisedException: If Quilr blocks the content.

        """
        verbose_proxy_logger.info("Quilr unified %s: received", input_type)

        texts = list(inputs.get("texts") or [])
        if not self.api_key or not texts:
            return inputs

        metadata = self._caller_metadata(request_data)
        if not self._matches_filters(request_data.get("model"), metadata):
            verbose_proxy_logger.info("Quilr unified %s: skipped (filter)", input_type)
            return inputs

        if input_type == "request":
            checked_texts = await self._check_request_texts(
                texts, inputs.get("structured_messages"), metadata
            )
        else:
            checked_texts = await self._check_response_texts(texts, metadata)

        # A failed Quilr call returns None: fail open and keep the original texts
        if checked_texts is None:
            return inputs

        inputs["texts"] = checked_texts
        return inputs

    async def _check_request_texts(
        self,
        texts: List[str],
        structured_messages: Optional[List[Any]],
        metadata: Dict[str, Any],
    ) -> Optional[List[str]]:
        """Check a request's user texts in one Quilr call.

        Args:
            texts: All texts LiteLLM extracted from the request
            structured_messages: Request messages with roles, if available
            metadata: Caller details sent to Quilr

        Returns:
            Checked texts (same length and order as ``texts``), or None if
            the Quilr call failed

        """
        # Quilr checks only user messages on the request side, so only those are sent
        indices = self._user_text_indices(texts, structured_messages)
        messages = [{"role": "user", "content": texts[i]} for i in indices]

        result = await self._post_to_quilr({"messages": messages}, metadata)
        if result is None:
            verbose_proxy_logger.info("Quilr unified request: error, passed through")
            return None

        self._raise_if_blocked(result, "request")

        checked_texts = list(texts)
        redacted_messages = result.get("messages")
        if (
            result.get("status") == "redacted"
            and isinstance(redacted_messages, list)
            and len(redacted_messages) == len(indices)
        ):
            # Quilr returns the messages in the order sent; put each one back at
            # the position its text came from
            for i, message in zip(indices, redacted_messages):
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    checked_texts[i] = content
            verbose_proxy_logger.info("Quilr unified request: redacted")

        verbose_proxy_logger.info("Quilr unified request: passed")
        return checked_texts

    async def _check_response_texts(
        self, texts: List[str], metadata: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Check each text of a response with Quilr.

        Args:
            texts: All texts LiteLLM extracted from the response
            metadata: Caller details sent to Quilr

        Returns:
            Checked texts (same length and order as ``texts``), or None if
            the Quilr call failed

        """
        checked_texts = list(texts)
        for i, text in enumerate(texts):
            if not text:
                continue

            result = await self._post_to_quilr(
                {"text": text, "type": "response"}, metadata
            )
            if result is None:
                verbose_proxy_logger.info(
                    "Quilr unified response: error, passed through"
                )
                return None

            self._raise_if_blocked(result, "response")

            if result.get("status") == "redacted" and result.get("processed_text"):
                checked_texts[i] = result["processed_text"]
                verbose_proxy_logger.info("Quilr unified response: redacted")

        verbose_proxy_logger.info("Quilr unified response: passed")
        return checked_texts

    def _user_text_indices(
        self, texts: List[str], structured_messages: Optional[List[Any]]
    ) -> List[int]:
        """Find which texts came from user messages.

        LiteLLM's ``texts`` list has no roles, so each text is matched against
        the user messages in ``structured_messages``.

        Args:
            texts: All texts LiteLLM extracted from the request
            structured_messages: Request messages with roles, if available

        Returns:
            Positions in ``texts`` that belong to user messages. Every
            position if there are no roles (e.g. Gemini) or nothing matches,
            so nothing is skipped when roles are unknown.

        """
        user_texts = set()
        for message in structured_messages or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                user_texts.add(content)
            elif isinstance(content, list):
                # Multi-part content, e.g. [{"type": "text", "text": "..."}, ...]
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        user_texts.add(part["text"])

        indices = [i for i, text in enumerate(texts) if text in user_texts]
        return indices or list(range(len(texts)))

    def _raise_if_blocked(self, result: Dict[str, Any], input_type: str) -> None:
        """Raise LiteLLM's guardrail exception if Quilr blocked the content.

        Args:
            result: Quilr API response
            input_type: "request" or "response", used in the log line

        Raises:
            GuardrailRaisedException: If ``result["status"]`` is "blocked".

        """
        if result.get("status") != "blocked":
            return

        verbose_proxy_logger.info("Quilr unified %s: blocked", input_type)
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=self._format_block_message(result.get("categories_detected", [])),
            should_wrap_with_default_message=False,
            blocked_content=True,
        )

    async def _post_to_quilr(
        self, payload: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Send a payload to Quilr's ``/sdk/v1/check`` endpoint.

        Same endpoint, headers and timeout as ``QuilrGuardrail._call_quilr_api``.
        Metadata is passed as a dict because LiteLLM does not pass a
        ``UserAPIKeyAuth`` object to ``apply_guardrail``.

        Args:
            payload: Request body ({"messages": [...]} or {"text": ..., "type": ...})
            metadata: Caller details added under ``payload["metadata"]``

        Returns:
            Quilr API response, or None on any error (timeout, connection,
            HTTP status or invalid response)

        """
        payload["metadata"] = {"caller": "litellm", **metadata}
        try:
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            response = await async_client.post(
                f"{self.api_base.rstrip('/')}/sdk/v1/check",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def _caller_metadata(self, request_data: dict) -> Dict[str, Any]:
        """Collect caller details (API key name, team, user, ...) for Quilr.

        LiteLLM stores them in the request metadata with a ``user_api_key_``
        prefix, which is removed here (e.g. ``user_api_key_key_name`` ->
        ``key_name``). Which fields are present depends on the LiteLLM route.

        Args:
            request_data: The original request data

        Returns:
            JSON-serializable caller details, without None values

        """
        source = (
            request_data.get("litellm_metadata") or request_data.get("metadata") or {}
        )
        prefix = "user_api_key_"
        result = {}
        for key, value in source.items():
            if not key.startswith(prefix) or value is None:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue
            result[key[len(prefix) :]] = value
        return result

    def _matches_filters(self, model: Optional[str], metadata: Dict[str, Any]) -> bool:
        """Apply the model and key-name filters, like QuilrGuardrail.

        Filters come from ``APPLY_QUILR_GUARDRAILS_FOR_MODELS`` and
        ``APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES``. LiteLLM does not include the
        key name in the metadata on every route (e.g. ``/v1/messages``); there,
        a key-name filter skips the request.

        Args:
            model: Model name from the request
            metadata: Caller details from ``_caller_metadata``

        Returns:
            True if the guardrail should run for this request

        """
        if self.allowed_models and (not model or model not in self.allowed_models):
            return False

        if self.allowed_key_names:
            key_name = metadata.get("key_name")
            if not key_name or key_name not in self.allowed_key_names:
                return False

        return True

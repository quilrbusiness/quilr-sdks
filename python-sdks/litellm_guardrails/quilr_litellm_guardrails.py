"""
Quilr Guardrails Integration for LiteLLM

This module provides a custom guardrail class that integrates Quilr's guardrails API
with LiteLLM proxy for both request-side and response-side content checking.

Environment Variables:
    QUILR_GUARDRAILS_KEY: API key for Quilr guardrails (required)
    QUILR_GUARDRAILS_BASE_URL: Base URL for Quilr guardrails API (default: https://guardrails.quilr.ai)
    APPLY_QUILR_GUARDRAILS_FOR_MODELS: Comma-separated list of models to apply guardrails to (optional, if not set applies to all)
    APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES: Comma-separated list of API key names to apply guardrails to (optional, if not set applies to all)

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

    curl -X POST http://localhost:4000/v1/chat/completions \\
      -H "Authorization: Bearer sk-xxx" \\
      -H "Content-Type: application/json" \\
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
"""

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


class QuilrGuardrail(CustomGuardrail):
    """
    Custom guardrail that integrates with Quilr's guardrails API.

    Handles both request-side (input) and response-side (output) checking:
    - Request-side: Checks messages before sending to LLM, can redact or block
    - Response-side: Checks LLM output before returning to user, can redact or block
    """

    def __init__(self, **kwargs):
        self.api_key = os.getenv("QUILR_GUARDRAILS_KEY")
        self.api_base = os.getenv("QUILR_GUARDRAILS_BASE_URL", "https://guardrails.quilr.ai")

        # Parse optional filters (comma-separated lists)
        models_env = os.getenv("APPLY_QUILR_GUARDRAILS_FOR_MODELS", "")
        self.allowed_models = [m.strip() for m in models_env.split(",") if m.strip()] or None

        key_names_env = os.getenv("APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES", "")
        self.allowed_key_names = [k.strip() for k in key_names_env.split(",") if k.strip()] or None

        if not self.api_key:
            verbose_proxy_logger.warning(
                "QUILR_GUARDRAILS_KEY not set. Quilr guardrail will not function."
            )

        super().__init__(**kwargs)

    async def _call_quilr_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to Quilr's guardrails API.

        Args:
            payload: Request body (either {"messages": [...]} or {"text": "..."})

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

        response = await async_client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=10,
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
        """
        Convert OpenAI Responses API input format to chat messages format.

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
        """
        Convert chat messages format back to OpenAI Responses API format.

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
        """
        Check if guardrail should be applied based on filters.

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
            "aresponses",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Check request content before sending to LLM.

        Supports both Chat Completions API (messages) and Responses API (input/instructions).

        - If blocked: raises exception
        - If redacted: replaces messages/input with redacted version
        - If safe: passes through unchanged
        """
        verbose_proxy_logger.info("Quilr pre-call: request received")

        if not self.api_key:
            return data

        # Check if guardrail should be applied based on filters
        if not self._should_apply_guardrail(data.get("model"), user_api_key_dict):
            verbose_proxy_logger.info("Quilr pre-call: skipped (filter)")
            return data

        # Determine if this is a Responses API call
        is_responses_api = call_type == "aresponses"

        if is_responses_api:
            # Handle OpenAI Responses API format
            input_data = data.get("input")
            instructions = data.get("instructions")

            if not input_data:
                return data

            original_input_was_string = isinstance(input_data, str)
            messages = self._responses_input_to_messages(input_data, instructions)
        else:
            # Handle Chat Completions API format
            messages = data.get("messages")
            if not messages:
                return data

        try:
            result = await self._call_quilr_api({"messages": messages})
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
                    if is_responses_api:
                        # Convert back to Responses API format
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
                    verbose_proxy_logger.info("Quilr pre-call: redacted")

            verbose_proxy_logger.info("Quilr pre-call: passed")
            return data

        except RejectedRequestError:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Quilr pre-call: error - %s", str(e))
            raise

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
            "aresponses",
        ],
    ):
        """
        Check request content in parallel with LLM call (during_call mode).

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

        # Check if guardrail should be applied based on filters
        if not self._should_apply_guardrail(data.get("model"), user_api_key_dict):
            verbose_proxy_logger.info("Quilr during-call: skipped (filter)")
            return

        # Determine if this is a Responses API call
        is_responses_api = call_type == "aresponses"

        if is_responses_api:
            # Handle OpenAI Responses API format
            input_data = data.get("input")
            instructions = data.get("instructions")

            if not input_data:
                return

            original_input_was_string = isinstance(input_data, str)
            messages = self._responses_input_to_messages(input_data, instructions)
        else:
            # Handle Chat Completions API format
            messages = data.get("messages")
            if not messages:
                return

        try:
            result = await self._call_quilr_api({"messages": messages})
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
                    if is_responses_api:
                        # Convert back to Responses API format
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
                    verbose_proxy_logger.info("Quilr during-call: redacted")

            verbose_proxy_logger.info("Quilr during-call: passed")

        except RejectedRequestError:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Quilr during-call: error - %s", str(e))
            raise

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Check LLM response before returning to user.

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
            return await self._check_responses_api_output(response, data)

        # Handle Chat Completions API format
        if isinstance(response, litellm.ModelResponse):
            return await self._check_chat_completions_output(response, data)

        return response

    async def _check_chat_completions_output(
        self, response: litellm.ModelResponse, data: dict
    ) -> litellm.ModelResponse:
        """Check output for Chat Completions API responses."""
        for choice in response.choices:
            if not hasattr(choice, "message") or not choice.message:
                continue

            content = choice.message.content
            if not content or not isinstance(content, str):
                continue

            try:
                result = await self._call_quilr_api({"text": content})
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
            except Exception as e:
                verbose_proxy_logger.error("Quilr post-call: error - %s", str(e))
                raise

        verbose_proxy_logger.info("Quilr post-call: passed")
        return response

    async def _check_responses_api_output(
        self, response: ResponsesAPIResponse, data: dict
    ) -> ResponsesAPIResponse:
        """Check output for OpenAI Responses API responses."""
        # Collect all output_text items and their text content
        output_text_items = []

        if response.output:
            for output_item in response.output:
                # Check if this is a message output (has content attribute)
                if hasattr(output_item, "content") and output_item.content:
                    for content_item in output_item.content:
                        # Check if this is an output_text item
                        if hasattr(content_item, "type") and content_item.type == "output_text":
                            if hasattr(content_item, "text") and content_item.text:
                                output_text_items.append(content_item)

        if not output_text_items:
            verbose_proxy_logger.info("Quilr post-call: no output text found")
            return response

        # Concatenate all text content
        all_text = "\n".join(item.text for item in output_text_items)

        try:
            result = await self._call_quilr_api({"text": all_text})
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
                    # Replace first output_text item with redacted text
                    output_text_items[0].text = processed_text
                    # Clear other output_text items
                    for item in output_text_items[1:]:
                        item.text = ""
                    verbose_proxy_logger.info("Quilr post-call: redacted")

        except RejectedRequestError:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Quilr post-call: error - %s", str(e))
            raise

        verbose_proxy_logger.info("Quilr post-call: passed")
        return response

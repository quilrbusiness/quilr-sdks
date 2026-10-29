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
    QUILR_GUARDRAILS_FAIL_CLOSED: QuilrUnifiedGuardrail only. "true" blocks
        content when the Quilr API can't be reached (default: false, passes)
    QUILR_GUARDRAILS_SCAN_MCP_TOOLS: QuilrUnifiedGuardrail only. Which MCP
        tools to check: "all" (default), "none", or a comma-separated mix of
        "readonly", "writeonly", "destructive" and tool names

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

    QuilrUnifiedGuardrail also checks tool-call arguments, MCP tool calls
    (modes pre_mcp_call, during_mcp_call, post_mcp_call; which tools is set
    with QUILR_GUARDRAILS_SCAN_MCP_TOOLS) and streamed answers,
    records Quilr's verdict in LiteLLM's logs (Langfuse, Datadog, ...) and can
    block when Quilr can't be reached. See the class docstring and README.
"""

import asyncio
import contextvars
import functools
import json
import os
import time
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


@functools.lru_cache(maxsize=1)
def _mcp_prefix_separator() -> str:
    """Return the separator LiteLLM puts between an MCP server and tool name."""
    try:
        from litellm.proxy._experimental.mcp_server.utils import (
            MCP_TOOL_PREFIX_SEPARATOR,
        )

        return MCP_TOOL_PREFIX_SEPARATOR
    except ImportError:
        return os.getenv("MCP_TOOL_PREFIX_SEPARATOR", "-")


def _annotation(tool: Any, camel: str, snake: str) -> Any:
    """Read an MCP tool annotation (dict, or object with snake_case or camelCase fields)."""
    annotations = getattr(tool, "annotations", None)
    if annotations is None:
        return None
    if isinstance(annotations, dict):
        return annotations.get(camel, annotations.get(snake))
    value = getattr(annotations, snake, None)
    return getattr(annotations, camel, None) if value is None else value


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


# State of one QuilrUnifiedGuardrail.apply_guardrail call: identity headers
# and trace details sent to Quilr, and the Quilr results for LiteLLM's logs
_QUILR_CALL_STATE: contextvars.ContextVar = contextvars.ContextVar(
    "quilr_call_state", default=None
)


class QuilrUnifiedGuardrail(QuilrGuardrail):
    """Quilr guardrail built on LiteLLM's unified ``apply_guardrail`` interface.

    Use this class to check API formats that ``QuilrGuardrail`` does not read,
    such as Anthropic ``/v1/messages`` and Gemini ``generateContent``.
    LiteLLM extracts the texts from each format it has a guardrail translator
    for, passes them to ``apply_guardrail``, and writes the checked texts back.

    Behavior (same Quilr endpoint and results as ``QuilrGuardrail``):
        - Request: user texts are checked. If LiteLLM gives no message roles
          (e.g. Gemini, embeddings), all texts are checked; several texts are
          checked together in one JSON-mode call.
        - Response: each text is checked.
        - Tool calls (request and response): the arguments of all tool calls
          are checked together in one call, using Quilr's JSON mode.
        - MCP tool calls (modes ``pre_mcp_call``, ``during_mcp_call``,
          ``post_mcp_call``): the tool description and argument values, or
          the tool result, are checked together in one JSON-mode call.
        - blocked: the call is rejected with an error
        - redacted: texts are replaced with Quilr's redacted versions
        - safe: texts pass through unchanged
        - Quilr call fails (timeout, connection or API error): texts pass
          through unchanged and a warning is logged (or the content is
          blocked with ``quilr_fail_closed``, see below)

    Streaming answers (mode ``post_call``): LiteLLM checks the answer so far
    every few chunks and once at the end. A block ends the stream with the
    block message. Redaction while streaming needs
    ``streaming_transform_mode: incremental_diff`` (OpenAI chat format only);
    the last ``stream_holdback_chars`` characters are then held back so a
    value found in them can still be redacted. Optional settings, under the
    guardrail's ``litellm_params``:

        streaming_sampling_rate: 5             # check every Nth chunk
        streaming_end_of_stream_only: false    # true: check once at the end
        streaming_buffer_until_moderated: false  # true: send nothing until checked
        streaming_buffer_release_on_scan: false  # with buffering: release after each check
        streaming_transform_mode: block_only   # or incremental_diff (redaction)
        stream_holdback_chars: 64              # incremental_diff only

    When Quilr can't be reached (timeout, connection or API error), content
    passes through unchecked and a warning is logged. To block it instead,
    set ``quilr_fail_closed: true`` under ``litellm_params`` or the
    environment variable ``QUILR_GUARDRAILS_FAIL_CLOSED=true``. (LiteLLM's own
    ``unreachable_fallback`` is not used: LiteLLM passes it as "fail_closed"
    even when it isn't configured, which would change the default.)

    MCP tools to check: by default every MCP tool is checked. To check only
    some, set ``quilr_scan_mcp_tools`` under ``litellm_params`` or the
    environment variable ``QUILR_GUARDRAILS_SCAN_MCP_TOOLS``:

        all                  # every tool (default, also when empty)
        none                 # no MCP tools
        readonly             # tools labeled read-only (MCP readOnlyHint)
        writeonly            # tools not labeled read-only
        destructive          # tools that may delete or overwrite (MCP destructiveHint)
        run_sql,read_logs    # these tools (plain or server-prefixed name)
        writeonly,run_sql    # a mix: a tool is checked if any item matches

    - Applies to all MCP modes and to the tool list check.
    - A tool without labels counts as writeonly and destructive (the MCP
      spec's defaults), so it is still checked by those.
    - Tool names are case-sensitive; label words are not and may be written
      with "-" or "_" ("read-only"). In config.yaml the value may be a list.
    - "none" together with other values is treated as "all" (with a warning).
    - The labels come from the MCP server's tool list, read through
      LiteLLM's MCP server manager and cached for ``_MCP_LABEL_TTL_SECONDS``
      (``_MCP_LABEL_RETRY_SECONDS`` after a failed read). If the list can't
      be read, the tool counts as unlabeled. This is always the case for
      servers that need each caller's own login (per-user token): there
      ``readonly`` checks nothing, so use tool names.
    - In ``during_mcp_call`` LiteLLM gives no server name, so the labels are
      looked up on every server; a name found on several servers counts as
      read-only only if it is read-only on all of them.
    - The labels are set by the MCP server and are not verified.
    - A skipped tool passes unchanged and is logged with status ``not_run``.

    Requires a LiteLLM version with ``apply_guardrail`` support
    (tested with 1.106.2).
    """

    # Streaming settings LiteLLM reads from the guardrail object
    _STREAMING_SETTINGS = (
        "streaming_sampling_rate",
        "streaming_end_of_stream_only",
        "streaming_buffer_until_moderated",
        "streaming_buffer_release_on_scan",
        "streaming_transform_mode",
    )
    _DEFAULT_STREAM_HOLDBACK_CHARS = 64
    _MCP_LABEL_TTL_SECONDS = 300
    _MCP_LABEL_RETRY_SECONDS = 30  # after a failed tool list

    def __init__(self, **kwargs):
        """Initialize like QuilrGuardrail and keep the streaming settings.

        LiteLLM passes the guardrail's extra ``litellm_params`` as keyword
        arguments but does not store them, while its streaming code reads
        them from the guardrail object. So they are set here.
        """
        streaming_settings = {
            name: kwargs.pop(name)
            for name in self._STREAMING_SETTINGS
            if name in kwargs
        }
        holdback = kwargs.pop("stream_holdback_chars", None)
        fail_closed = kwargs.pop(
            "quilr_fail_closed", os.getenv("QUILR_GUARDRAILS_FAIL_CLOSED", "false")
        )
        scan_mcp_tools = kwargs.pop(
            "quilr_scan_mcp_tools", os.getenv("QUILR_GUARDRAILS_SCAN_MCP_TOOLS", "")
        )
        super().__init__(**kwargs)

        self.fail_closed = str(fail_closed).strip().lower() in ("1", "true", "yes")
        self.scan_mcp_tools = self._parse_scan_mcp_tools(scan_mcp_tools)
        # server ID -> (expiry time, {tool name: (read_only, destructive)})
        self._mcp_label_cache: Dict[str, Any] = {}
        # server ID -> tool list fetch in progress, shared by concurrent checks
        self._mcp_label_fetches: Dict[str, Any] = {}

        for name, value in streaming_settings.items():
            setattr(self, name, value)
        if holdback is not None:
            self.stream_holdback_chars = max(0, int(holdback))
        elif streaming_settings.get("streaming_transform_mode") == "incremental_diff":
            self.stream_holdback_chars = self._DEFAULT_STREAM_HOLDBACK_CHARS
        else:
            self.stream_holdback_chars = 0

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
            inputs: Extracted content. Uses ``texts`` (list of strings),
                ``tool_calls`` (tool calls with JSON arguments) and, for
                requests, ``structured_messages`` (messages with roles)
            request_data: The original request data (model, metadata, ...)
            input_type: "request" for the prompt, "response" for the answer
            logging_obj: LiteLLM logging object (unused)

        Returns:
            ``inputs`` with ``texts`` and ``tool_calls`` replaced by the
            checked versions. LiteLLM requires both lists to keep the same
            length and order.

        Raises:
            GuardrailRaisedException: If Quilr blocks the content.

        """
        verbose_proxy_logger.info("Quilr unified %s: received", input_type)

        texts = list(inputs.get("texts") or [])
        tool_calls = list(inputs.get("tool_calls") or [])
        if not self.api_key or not (texts or tool_calls):
            return inputs

        metadata = self._caller_metadata(request_data)
        if not self._matches_filters(request_data.get("model"), metadata):
            verbose_proxy_logger.info("Quilr unified %s: skipped (filter)", input_type)
            return inputs

        mcp_tool_name = self._mcp_tool_name(request_data)
        if mcp_tool_name is not None and not await self._should_scan_mcp_tool(
            mcp_tool_name, request_data
        ):
            verbose_proxy_logger.info(
                "Quilr unified %s: skipped MCP tool %s (scan_mcp_tools)",
                input_type,
                mcp_tool_name,
            )
            self._log_mcp_tool_skipped(request_data)
            return inputs

        streaming = input_type == "response" and bool(request_data.get("stream"))
        state = {**self._trace_context(request_data), "results": []}
        token = _QUILR_CALL_STATE.set(state)
        start_time = time.time()
        try:
            await self._check_inputs(
                inputs, texts, tool_calls, input_type, request_data, metadata
            )
        except GuardrailRaisedException as error:
            # A plain exception cuts a stream off with an error; LiteLLM ends
            # the stream cleanly with the block message for this exception
            if streaming and hasattr(self, "raise_passthrough_exception"):
                self.raise_passthrough_exception(
                    violation_message=str(error), request_data=request_data
                )
            raise
        finally:
            _QUILR_CALL_STATE.reset(token)
            self._log_quilr_results(state["results"], request_data, start_time)

        if streaming and self.stream_holdback_chars and inputs.get("texts"):
            # Hold back the newest characters so a value found in them can
            # still be redacted (used by streaming_transform_mode incremental_diff)
            inputs["stream_holdback_chars"] = [self.stream_holdback_chars] * len(
                inputs["texts"]
            )
        return inputs

    @staticmethod
    def _trace_context(request_data: dict) -> Dict[str, Any]:
        """Collect the session, trace and user details to send to Quilr.

        The session ID goes in the ``X-Conversation-Id`` header (Quilr groups
        checks into conversations by it), the user's email in
        ``X-User-Email``, and the session and trace IDs in the metadata, so
        Quilr logs can be matched with LiteLLM / Langfuse traces.

        Args:
            request_data: The request data LiteLLM passed to ``apply_guardrail``

        Returns:
            ``{"headers": {...}, "trace": {...}}``

        """
        # LiteLLM keeps request metadata in "metadata" (the caller's) and/or
        # "litellm_metadata" (its own), depending on the route
        metadata: Dict[str, Any] = {}
        for key in ("metadata", "litellm_metadata"):
            if isinstance(request_data.get(key), dict):
                metadata.update(request_data[key])

        headers = {}
        trace = {}
        session_id = metadata.get("session_id") or request_data.get(
            "litellm_session_id"
        )
        if session_id:
            headers["X-Conversation-Id"] = str(session_id)
            trace["session_id"] = str(session_id)
        trace_id = metadata.get("trace_id") or request_data.get("litellm_trace_id")
        if trace_id:
            trace["trace_id"] = str(trace_id)

        # The first value that looks like an email: Langfuse trace user,
        # end user, then the owner of the LiteLLM key
        for candidate in (
            metadata.get("trace_user_id"),
            request_data.get("user"),
            metadata.get("user_api_key_end_user_id"),
            metadata.get("user_api_key_user_email"),
        ):
            if isinstance(candidate, str) and "@" in candidate:
                headers["X-User-Email"] = candidate.strip()
                break

        return {"headers": headers, "trace": trace}

    def _log_quilr_results(
        self,
        results: List[Optional[Dict[str, Any]]],
        request_data: dict,
        start_time: float,
    ) -> None:
        """Record Quilr's verdict in LiteLLM's guardrail logs.

        LiteLLM sends this record to its logging integrations (Langfuse,
        Datadog, spend logs, ...). It carries Quilr's status, action and
        categories, never the checked content.

        Args:
            results: Quilr responses of this call (None for a failed call)
            request_data: The request data, where LiteLLM keeps the record
            start_time: When the checks started (``time.time()``)

        """
        if not results:
            return

        checked = [result for result in results if isinstance(result, dict)]
        statuses = [result.get("status", "safe") for result in checked]
        categories = sorted(
            {
                category
                for result in checked
                for category in result.get("categories_detected") or []
                if isinstance(category, str)
            }
        )
        masked_entity_count: Dict[str, int] = {}
        for result in checked:
            if result.get("status") == "redacted":
                for category in set(result.get("categories_detected") or []):
                    masked_entity_count[category] = (
                        masked_entity_count.get(category, 0) + 1
                    )

        if "blocked" in statuses:
            quilr_status, action, status = "blocked", "block", "guardrail_intervened"
        elif len(checked) < len(results):
            # Quilr was unreachable for at least one check: content passed unchecked
            quilr_status, action, status = (
                "error",
                "allow",
                "guardrail_failed_to_respond",
            )
        elif "redacted" in statuses:
            quilr_status, action, status = "redacted", "redact", "success"
        elif categories:
            quilr_status, action, status = "safe", "allow", "guardrail_flagged"
        else:
            quilr_status, action, status = "safe", "allow", "success"

        end_time = time.time()
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={
                    "quilr_status": quilr_status,
                    "action": action,
                    "categories_detected": categories,
                    "checks": len(results),
                },
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                masked_entity_count=masked_entity_count or None,
                guardrail_provider="quilr",
                tracing_detail={
                    "violation_categories": categories,
                    "guardrail_action": action,
                },
            )
        except Exception as error:
            verbose_proxy_logger.debug("Quilr unified: log record failed: %s", error)

    async def _check_inputs(
        self,
        inputs: Dict[str, Any],
        texts: List[str],
        tool_calls: List[Any],
        input_type: str,
        request_data: dict,
        metadata: Dict[str, Any],
    ) -> None:
        """Check the texts and tool calls and write the checked versions to ``inputs``.

        Args:
            inputs: The inputs from LiteLLM, updated in place
            texts: Texts to check
            tool_calls: Tool calls to check
            input_type: "request" or "response"
            request_data: The original request data
            metadata: Caller details sent to Quilr

        Raises:
            GuardrailRaisedException: If Quilr blocks the content.

        """
        mcp_tool_name = self._mcp_tool_name(request_data)
        if texts and mcp_tool_name is not None:
            # MCP texts are separate values (description, argument values,
            # result parts), not chat messages: check all of them as JSON
            checked_texts = await self._check_mcp_texts(
                texts, input_type, mcp_tool_name, request_data, metadata
            )
            if checked_texts is not None:
                inputs["texts"] = checked_texts
        elif texts:
            if (
                input_type == "request"
                and not inputs.get("structured_messages")
                and len(texts) > 1
            ):
                # Separate texts without roles (e.g. an embeddings batch):
                # Quilr's message mode would check only the last ones
                checked_texts = await self._check_texts_as_json(
                    texts,
                    input_type,
                    {**metadata, "scan_context": "request_texts"},
                    "request texts",
                )
            elif input_type == "request":
                checked_texts = await self._check_request_texts(
                    texts, inputs.get("structured_messages"), metadata
                )
            else:
                checked_texts = await self._check_response_texts(texts, metadata)

            # A failed Quilr call returns None: fail open and keep the original texts
            if checked_texts is not None:
                inputs["texts"] = checked_texts

        if tool_calls:
            checked_tool_calls = await self._check_tool_calls(
                tool_calls, input_type, metadata
            )
            if checked_tool_calls is not None:
                inputs["tool_calls"] = checked_tool_calls

    @staticmethod
    def _mcp_tool_name(request_data: dict) -> Optional[str]:
        """Return the MCP tool name if this check is for an MCP tool call.

        LiteLLM marks MCP checks differently per stage: the pre/during-call
        request has ``mcp_tool_name``; the post-call request data has
        ``mcp_tool_call_metadata`` or the ``call_mcp_tool`` call type.

        Args:
            request_data: The request data LiteLLM passed to ``apply_guardrail``

        Returns:
            The tool name ("unknown" if not given), or None if not MCP

        """
        if request_data.get("mcp_tool_name"):
            return str(request_data["mcp_tool_name"])
        tool_metadata = request_data.get("mcp_tool_call_metadata")
        if isinstance(tool_metadata, dict) and tool_metadata.get("name"):
            return str(tool_metadata["name"])
        if request_data.get("call_type") in ("call_mcp_tool", "list_mcp_tools"):
            return str(request_data.get("name") or "unknown")
        return None

    @staticmethod
    def _mcp_server_name(request_data: dict) -> Optional[str]:
        """Return the MCP server name of this check, if LiteLLM gives it.

        The pre-call request has ``mcp_server_name``; the post-call request
        has it in ``mcp_tool_call_metadata``; the during-call request has none.
        """
        if request_data.get("mcp_server_name"):
            return str(request_data["mcp_server_name"])
        tool_metadata = request_data.get("mcp_tool_call_metadata")
        if isinstance(tool_metadata, dict) and tool_metadata.get("mcp_server_name"):
            return str(tool_metadata["mcp_server_name"])
        return None

    # Label values of scan_mcp_tools (anything else is a tool name)
    _MCP_TOOL_LABELS = ("readonly", "writeonly", "destructive")

    @staticmethod
    def _parse_scan_mcp_tools(value: Any) -> Optional[Dict[str, Any]]:
        """Parse the ``scan_mcp_tools`` setting.

        Args:
            value: "all", "none", or a comma-separated mix (or YAML list) of
                "readonly", "writeonly", "destructive" and tool names. Label
                spellings with "-" or "_" (e.g. "read-only") are accepted.

        Returns:
            None to check every MCP tool, otherwise
            ``{"labels": {label values}, "names": {tool names}}``
            (both empty for "none")

        """
        if isinstance(value, (list, tuple, set)):
            raw_items = [str(item) for item in value]
        else:
            raw_items = str(value or "").split(",")
        items = [item.strip() for item in raw_items if item.strip()]

        def keyword(item: str) -> str:
            return item.lower().replace("-", "").replace("_", "").replace(" ", "")

        keywords = {keyword(item) for item in items}
        if not items or "all" in keywords:
            return None
        if "none" in keywords:
            if len(items) > 1:
                verbose_proxy_logger.warning(
                    "Quilr unified: scan_mcp_tools %r mixes 'none' with other "
                    "values; checking all MCP tools",
                    value,
                )
                return None
            return {"labels": set(), "names": set()}
        labels = keywords & set(QuilrUnifiedGuardrail._MCP_TOOL_LABELS)
        names = {
            item
            for item in items
            if keyword(item) not in QuilrUnifiedGuardrail._MCP_TOOL_LABELS
        }
        verbose_proxy_logger.info(
            "Quilr unified: checking MCP tools with labels %s and names %s",
            sorted(labels),
            sorted(names),
        )
        return {"labels": labels, "names": names}

    async def _should_scan_mcp_tool(self, tool_name: str, request_data: dict) -> bool:
        """Return True if this MCP tool should be checked (``scan_mcp_tools``).

        Args:
            tool_name: The tool name LiteLLM passed (without the server prefix)
            request_data: The request data, for the MCP server name

        Returns:
            True if any item of the setting matches the tool

        """
        setting = self.scan_mcp_tools
        if setting is None:
            return True
        if self._mcp_tool_listed(tool_name, setting["names"]):
            return True
        labels = setting["labels"]
        if not labels:
            return False
        if {"readonly", "writeonly"} <= labels:
            # every tool is either read-only or not
            return True
        read_only, destructive = await self._mcp_tool_labels(
            tool_name, self._mcp_server_name(request_data)
        )
        return (
            ("readonly" in labels and read_only)
            or ("writeonly" in labels and not read_only)
            or ("destructive" in labels and destructive)
        )

    @staticmethod
    def _mcp_tool_listed(tool_name: str, names: set) -> bool:
        """Return True if the tool is in the configured names.

        A name matches as given (``run_sql``) or with any server prefix
        (``myserver-run_sql``). The prefix is not used to tell servers apart:
        LiteLLM may prefix with the server's name, alias or ID, and a wrong
        guess would skip a tool that should be checked.
        """
        if tool_name in names:
            return True
        suffix = f"{_mcp_prefix_separator()}{tool_name}"
        return any(name.endswith(suffix) for name in names)

    async def _mcp_tool_labels(
        self, tool_name: str, server_name: Optional[str]
    ) -> tuple:
        """Return the MCP tool's labels as ``(read_only, destructive)``.

        LiteLLM doesn't pass the tool's labels to guardrails, so they are
        read from the server's tool list through LiteLLM's MCP server
        manager. A tool whose labels can't be read counts as not read-only
        and destructive, so ``writeonly`` and ``destructive`` still check it.

        Args:
            tool_name: The tool name without the server prefix
            server_name: The MCP server name, or None to look in all servers

        Returns:
            read_only: True only if every matching tool is labeled read-only
            destructive: True if any matching tool is destructive

        """
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager as manager,
            )

            if server_name:
                server = manager.get_mcp_server_by_name(server_name)
                servers = [server] if server is not None else []
            else:
                servers = list(manager.get_registry().values())
        except Exception as error:
            verbose_proxy_logger.warning(
                "Quilr unified: MCP tool labels unavailable: %s", error
            )
            return False, True

        # All servers at once, so a slow or unreachable server delays the
        # check by at most one timeout
        server_labels = await asyncio.gather(
            *(self._mcp_server_labels(manager, server) for server in servers)
        )
        found = [labels[tool_name] for labels in server_labels if tool_name in labels]
        if not found:
            return False, True
        return (
            all(read_only for read_only, _ in found),
            any(destructive for _, destructive in found),
        )

    async def _mcp_server_labels(self, manager: Any, server: Any) -> Dict[str, tuple]:
        """Return ``{tool name: (read_only, destructive)}`` for an MCP server, cached.

        Concurrent checks on the same server share one tool list fetch.

        Args:
            manager: LiteLLM's MCP server manager
            server: The MCP server

        Returns:
            The server's tools (without prefix) and their labels; empty if
            the tool list can't be read

        """
        key = str(getattr(server, "server_id", None) or getattr(server, "name", ""))
        cached = self._mcp_label_cache.get(key)
        if cached and cached[0] > time.time():
            return cached[1]

        fetch = self._mcp_label_fetches.get(key)
        if fetch is None:
            fetch = asyncio.ensure_future(
                self._fetch_mcp_server_labels(manager, server, key)
            )
            self._mcp_label_fetches[key] = fetch
            fetch.add_done_callback(lambda _: self._mcp_label_fetches.pop(key, None))
        return await asyncio.shield(fetch)

    async def _fetch_mcp_server_labels(
        self, manager: Any, server: Any, key: str
    ) -> Dict[str, tuple]:
        """Read an MCP server's tool list and cache each tool's labels.

        Follows the MCP spec's defaults: a tool is read-only only if
        ``readOnlyHint`` is true, and a tool that isn't read-only is
        destructive unless ``destructiveHint`` is false. A failed fetch is
        cached for ``_MCP_LABEL_RETRY_SECONDS`` only.
        """
        labels: Dict[str, tuple] = {}
        ttl = self._MCP_LABEL_TTL_SECONDS
        try:
            tools = await asyncio.wait_for(
                manager.get_tools_from_server(server, add_prefix=False),
                timeout=self.timeout,
            )
            for tool in tools:
                read_only_hint = _annotation(tool, "readOnlyHint", "read_only_hint")
                destructive_hint = _annotation(
                    tool, "destructiveHint", "destructive_hint"
                )
                read_only = read_only_hint is True
                destructive = not read_only and destructive_hint is not False
                labels[str(tool.name)] = (read_only, destructive)
        except Exception as error:
            ttl = self._MCP_LABEL_RETRY_SECONDS
            verbose_proxy_logger.warning(
                "Quilr unified: could not read MCP tool labels of %s: %s", key, error
            )
        self._mcp_label_cache[key] = (time.time() + ttl, labels)
        return labels

    def _log_mcp_tool_skipped(self, request_data: dict) -> None:
        """Record in LiteLLM's guardrail logs that an MCP tool was not checked."""
        now = time.time()
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={
                    "quilr_status": "skipped",
                    "action": "allow",
                    "reason": "scan_mcp_tools",
                },
                request_data=request_data,
                guardrail_status="not_run",
                start_time=now,
                end_time=now,
                duration=0.0,
                guardrail_provider="quilr",
            )
        except Exception as error:
            verbose_proxy_logger.debug("Quilr unified: log record failed: %s", error)

    async def _check_mcp_texts(
        self,
        texts: List[str],
        input_type: str,
        tool_name: str,
        request_data: dict,
        metadata: Dict[str, Any],
    ) -> Optional[List[str]]:
        """Check all texts of an MCP tool call or result in one Quilr JSON call.

        Args:
            texts: Description and argument values (request) or result
                texts (response), as extracted by LiteLLM
            input_type: "request" (before/while the tool runs) or "response"
                (the tool result)
            tool_name: MCP tool name, sent as metadata
            request_data: The request data (for the MCP server name)
            metadata: Caller details sent to Quilr

        Returns:
            Checked texts (same length and order), or None if the Quilr call
            failed or nothing changed

        Raises:
            GuardrailRaisedException: If Quilr blocks the content.

        """
        mcp_metadata = {
            **metadata,
            "scan_context": "mcp_tool_result"
            if input_type == "response"
            else "mcp_tool_call",
            "tool_names": [tool_name],
        }
        server_name = self._mcp_server_name(request_data)
        if server_name:
            mcp_metadata["mcp_server_name"] = server_name

        return await self._check_texts_as_json(
            texts, input_type, mcp_metadata, f"mcp {input_type}"
        )

    async def _check_texts_as_json(
        self,
        texts: List[str],
        input_type: str,
        metadata: Dict[str, Any],
        label: str,
    ) -> Optional[List[str]]:
        """Check separate texts in one Quilr call, as a JSON list.

        JSON mode checks every value, while Quilr's message mode checks only
        the last user messages (about 15 words), which would skip most of
        them.

        Args:
            texts: Texts to check
            input_type: "request" or "response"
            metadata: Caller details sent to Quilr
            label: Name of the check in log lines

        Returns:
            Checked texts (same length and order), or None if the Quilr call
            failed or nothing changed

        Raises:
            GuardrailRaisedException: If Quilr blocks the content.

        """
        result = await self._post_to_quilr(
            {"json": texts, "type": input_type}, metadata
        )
        if result is None:
            verbose_proxy_logger.info("Quilr unified %s: error, passed through", label)
            return None

        self._raise_if_blocked(result, label)

        processed = result.get("processed_json")
        if (
            result.get("status") != "redacted"
            or not isinstance(processed, list)
            or len(processed) != len(texts)
        ):
            verbose_proxy_logger.info("Quilr unified %s: passed", label)
            return None

        verbose_proxy_logger.info("Quilr unified %s: redacted", label)
        return [
            redacted if isinstance(redacted, str) else original
            for original, redacted in zip(texts, processed)
        ]

    async def _check_tool_calls(
        self,
        tool_calls: List[Any],
        input_type: str,
        metadata: Dict[str, Any],
    ) -> Optional[List[Any]]:
        """Check the arguments of all tool calls in one Quilr JSON call.

        Each tool call's arguments (a JSON string) are parsed and sent as one
        item of a JSON list. Quilr returns ``processed_json`` with the same
        structure and sensitive values redacted. Arguments that are not valid
        JSON are sent as a plain string item.

        Args:
            tool_calls: Tool calls from LiteLLM (dicts or objects with
                ``function.name`` and ``function.arguments``)
            input_type: "request" or "response"
            metadata: Caller details sent to Quilr

        Returns:
            Tool calls (same length and order) with redacted arguments, or
            None if nothing was checked or the Quilr call failed

        Raises:
            GuardrailRaisedException: If Quilr blocks the arguments.

        """
        indices = []
        arguments = []
        is_json = []
        names = []
        for i, tool_call in enumerate(tool_calls):
            name, raw_arguments = self._tool_call_function(tool_call)
            if not isinstance(raw_arguments, str) or not raw_arguments.strip():
                continue
            try:
                arguments.append(json.loads(raw_arguments))
                is_json.append(True)
            except ValueError:
                arguments.append(raw_arguments)
                is_json.append(False)
            indices.append(i)
            names.append(name or "unknown")

        if not indices:
            return None

        result = await self._post_to_quilr(
            {"json": arguments, "type": input_type},
            {**metadata, "scan_context": "tool_call_arguments", "tool_names": names},
        )
        if result is None:
            verbose_proxy_logger.info(
                "Quilr unified %s tool calls: error, passed through", input_type
            )
            return None

        self._raise_if_blocked(result, f"{input_type} tool calls")

        processed = result.get("processed_json")
        if (
            result.get("status") != "redacted"
            or not isinstance(processed, list)
            or len(processed) != len(indices)
        ):
            verbose_proxy_logger.info("Quilr unified %s tool calls: passed", input_type)
            return None

        checked_tool_calls = list(tool_calls)
        for i, original, redacted, was_json in zip(
            indices, arguments, processed, is_json
        ):
            if redacted == original:
                continue
            new_arguments = (
                json.dumps(redacted, ensure_ascii=False)
                if was_json or not isinstance(redacted, str)
                else redacted
            )
            checked_tool_calls[i] = self._with_tool_call_arguments(
                tool_calls[i], new_arguments
            )
        verbose_proxy_logger.info("Quilr unified %s tool calls: redacted", input_type)
        return checked_tool_calls

    @staticmethod
    def _tool_call_function(tool_call: Any) -> tuple:
        """Return ``(name, arguments)`` of a tool call dict or object."""
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        if isinstance(function, dict):
            return function.get("name"), function.get("arguments")
        return getattr(function, "name", None), getattr(function, "arguments", None)

    @staticmethod
    def _with_tool_call_arguments(tool_call: Any, arguments: str) -> Any:
        """Return the tool call with new arguments.

        Dicts are copied (LiteLLM writes the returned dict back into the
        request or response); objects are updated in place.
        """
        if isinstance(tool_call, dict):
            function = dict(tool_call.get("function") or {})
            function["arguments"] = arguments
            return {**tool_call, "function": function}
        function = getattr(tool_call, "function", None)
        if function is not None:
            function.arguments = arguments
        return tool_call

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
        state = _QUILR_CALL_STATE.get()
        identity_headers = dict(state["headers"]) if state else {}
        payload["metadata"] = {
            "caller": "litellm",
            **(state["trace"] if state else {}),
            **metadata,
        }
        try:
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )

            async def send():
                return await async_client.post(
                    f"{self.api_base.rstrip('/')}/sdk/v1/check",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        **identity_headers,
                    },
                    json=payload,
                    timeout=self.timeout,
                )

            try:
                response = await send()
            except Exception as error:
                # LiteLLM's HTTP client raises on error statuses. Quilr rejects
                # an email outside the app's allowed domains with a 401: check
                # without it rather than let the content pass unchecked
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status != 401 or "X-User-Email" not in identity_headers:
                    raise
                del identity_headers["X-User-Email"]
                response = await send()
            response.raise_for_status()
            result = response.json()
        except Exception as error:
            result = None
            status = getattr(getattr(error, "response", None), "status_code", None)
            reason = f"HTTP {status}" if status else type(error).__name__
            verbose_proxy_logger.warning(
                "Quilr unified: Quilr check failed (%s); %s",
                reason,
                "content blocked (fail-closed)"
                if self.fail_closed
                else "content passed unchecked (fail-open)",
            )

        if state is not None:
            state["results"].append(result)
        if result is None and self.fail_closed:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="Content blocked: the Quilr guardrail check could not be completed",
                should_wrap_with_default_message=False,
                status_code=503,
            )
        return result

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

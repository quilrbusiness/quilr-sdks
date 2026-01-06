"""
Quilr Guardrails Integration for LiteLLM

This module provides a custom guardrail class that integrates Quilr's guardrails API
with LiteLLM proxy for both request-side and response-side content checking.

Environment Variables:
    QUILR_GUARDRAILS_KEY: API key for Quilr guardrails (required)
    QUILR_GUARDRAILS_BASE_URL: Base URL for Quilr guardrails API (default: https://guardrails.quilr.ai)

Usage in LiteLLM config.yaml:

    guardrails:
      - guardrail_name: "quilr-input"
        litellm_params:
          guardrail: quilr_litellm_guardrails.QuilrGuardrail
          mode: "pre_call"              # checks input before LLM call

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

        verbose_proxy_logger.debug(
            "Quilr guardrail: calling %s with payload keys: %s",
            endpoint,
            list(payload.keys())
        )

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
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Check request content before sending to LLM.

        - If blocked: raises exception
        - If redacted: replaces messages with redacted version
        - If safe: passes through unchanged
        """
        if not self.api_key:
            verbose_proxy_logger.warning("Quilr guardrail: No API key, skipping pre-call check")
            return data

        messages = data.get("messages")
        if not messages:
            return data

        try:
            result = await self._call_quilr_api({"messages": messages})

            status = result.get("status", "safe")
            categories = result.get("categories_detected", [])

            verbose_proxy_logger.debug(
                "Quilr guardrail pre-call: status=%s, categories=%s",
                status,
                categories
            )

            if status == "blocked":
                raise RejectedRequestError(
                    message=self._format_block_message(categories),
                    model=data.get("model", ""),
                    llm_provider="quilr_guardrail",
                    request_data=data,
                )

            if status == "redacted":
                redacted_messages = result.get("messages")
                if redacted_messages:
                    data["messages"] = redacted_messages
                    verbose_proxy_logger.info(
                        "Quilr guardrail: Request messages redacted, categories: %s",
                        categories
                    )

            return data

        except RejectedRequestError:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Quilr guardrail pre-call error: %s", str(e))
            raise

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Check LLM response before returning to user.

        - If blocked: raises exception
        - If redacted: modifies response content with redacted version
        - If safe: passes through unchanged
        """
        if not self.api_key:
            verbose_proxy_logger.warning("Quilr guardrail: No API key, skipping post-call check")
            return response

        if not isinstance(response, litellm.ModelResponse):
            return response

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

                verbose_proxy_logger.debug(
                    "Quilr guardrail post-call: status=%s, categories=%s",
                    status,
                    categories
                )

                if status == "blocked":
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
                        verbose_proxy_logger.info(
                            "Quilr guardrail: Response redacted, categories: %s",
                            categories
                        )

            except RejectedRequestError:
                raise
            except Exception as e:
                verbose_proxy_logger.error("Quilr guardrail post-call error: %s", str(e))
                raise

        return response

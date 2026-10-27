## Using Quilr Guardrails with Litellm

### Prerequisites

- Quilr API Key from Quilr guardrails (from either self hosted Quilr guardrails or Quilr hosted)
- Download [quilr_litellm_guardrails.py](https://github.com/quilrbusiness/quilr-sdks/blob/main/python-sdks/litellm_guardrails/quilr_litellm_guardrails.py) and place it in the same directory as your litellm config.yaml file

### Using with litellm

1. Set the following environment variables

```bash
QUILR_GUARDRAILS_KEY=sk-quilr-XXXXXXXXX
QUILR_GUARDRAILS_BASE_URL=QUILR GUARDRAILS BASE URL
QUILR_GUARDRAILS_TIMEOUT=3  # Optional: timeout in seconds (default: 3). If timeout or API error occurs, request passes through.
QUILR_GUARDRAILS_FAIL_CLOSED=false  # Optional, QuilrUnifiedGuardrail only: true blocks content when Quilr can't be reached (see below)
```

2. Add guardrails to your litellm `config.yaml` file. Choose the modes you need:

| Mode | When it runs | What it checks | Pros | Cons |
|------|-------------|----------------|------|------|
| `pre_call` | Before LLM call (sequential) | Input | Can block malicious requests before they reach the LLM, prevents data leakage | Adds minimal latency |
| `during_call` | In parallel with LLM call | Input | No added latency | Cannot prevent data leakage or attacks (LLM processes request before guardrail completes). |
| `post_call` | After LLM call | Output | Can check LLM responses for policy violations | Adds guardrail latency. Only needed if response needs to be checked with guardrails |

**Example: Input guardrail only (pre_call)**
```yaml
guardrails:
  - guardrail_name: "quilr-input"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrGuardrail
      mode: "pre_call"
      default_on: true
```

**Example: Input guardrail with lower latency (during_call)**
```yaml
guardrails:
  - guardrail_name: "quilr-input-duringcall"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrGuardrail
      mode: "during_call"
      default_on: true
```

**Example: Output guardrail only (post_call)**
```yaml
guardrails:
  - guardrail_name: "quilr-output"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrGuardrail
      mode: "post_call"
      default_on: true
```

**Example: Both input and output guardrails**
```yaml
guardrails:
  - guardrail_name: "quilr-input"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrGuardrail
      mode: "pre_call"
      default_on: true

  - guardrail_name: "quilr-output"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrGuardrail
      mode: "post_call"
      default_on: true
```

**When to use `during_call` vs `pre_call`:**
- Use `during_call` for better latency - guardrail runs concurrently with LLM
- Use `pre_call` if you want to avoid wasting LLM compute on blocked requests

### Checking all API formats (`QuilrUnifiedGuardrail`)

`QuilrGuardrail` checks the OpenAI Chat Completions (`/v1/chat/completions`) and Responses (`/v1/responses`) formats. Other formats are checked only partly or not at all, for example Anthropic `/v1/messages` (the response is not checked) and Gemini `generateContent` (neither the request nor the response is checked).

To check these formats too, use `QuilrUnifiedGuardrail` instead. It works with all the modes and examples above. Only the `guardrail` line changes:

```yaml
guardrails:
  - guardrail_name: "quilr-input"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrUnifiedGuardrail
      mode: "pre_call"
      default_on: true

  - guardrail_name: "quilr-output"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrUnifiedGuardrail
      mode: "post_call"
      default_on: true
```

- LiteLLM extracts the text from each format it has a guardrail translator for and passes it to Quilr, so the same Quilr checks run for those formats.
- Requests: only user messages are checked, the same as `QuilrGuardrail`. For formats where LiteLLM gives no message roles (e.g. Gemini), all request text is checked.
- Requests with several separate texts and no message roles (e.g. an embeddings batch) are checked together in one call, so every text is checked.
- Blocking, redaction, the model filter and the timeout behave the same as `QuilrGuardrail`.
- Key-name filter: LiteLLM does not pass the key name on every route (e.g. Anthropic `/v1/messages`). On those routes, if `APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES` is set, requests are not checked.
- Requires a LiteLLM version with `apply_guardrail` support (tested with 1.106.2).

Supported through LiteLLM (tested): Chat Completions, Responses, Anthropic `/v1/messages`, Gemini `generateContent`, text completions, embeddings (input), image generation (prompt) and rerank (query). Other endpoints LiteLLM has a guardrail translator for use the same path.

The sections below apply to `QuilrUnifiedGuardrail` only.

#### Tool calls

Tool calls are checked automatically in the `pre_call`, `during_call` and `post_call` modes, both the tool calls the model asks to run and earlier tool calls in the conversation. The arguments of all tool calls are sent to Quilr together, as JSON:

- blocked: the request or response is rejected
- redacted: the sensitive values in the arguments are replaced, for example `{"body": "my SSN is <qe-ssn>"}`

For Anthropic `/v1/messages` responses, LiteLLM doesn't apply changed tool calls, so redaction doesn't apply there (blocking does).

#### MCP tool calls

When LiteLLM is used as an MCP gateway, add guardrails with the MCP modes:

| Mode | When it runs | What it checks | Can block | Can redact |
|------|-------------|----------------|-----------|------------|
| `pre_mcp_call` | Before the tool runs, and when clients list tools | Tool description and argument values | Yes | Yes (arguments) |
| `during_mcp_call` | In parallel with the tool | Argument values | Yes | No |
| `post_mcp_call` | After the tool runs, before the result reaches the model | Tool result | Yes | Yes (result text) |

```yaml
guardrails:
  - guardrail_name: "quilr-mcp-input"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrUnifiedGuardrail
      mode: "pre_mcp_call"
      default_on: true

  - guardrail_name: "quilr-mcp-output"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrUnifiedGuardrail
      mode: "post_mcp_call"
      default_on: true
```

- Works on all LiteLLM MCP routes: `/mcp` (Streamable HTTP), `/mcp-rest/tools/call` and tool search.
- `pre_mcp_call` also runs when a client lists tools: one Quilr call per tool.
- If Quilr redacts a field of a structured tool result that can't be changed (a key or a number), LiteLLM blocks the result instead.
- The model filter (`APPLY_QUILR_GUARDRAILS_FOR_MODELS`) skips MCP checks, since LiteLLM gives MCP calls the model name `mcp-tool-call`.

#### Streaming responses

With `post_call`, streamed answers are checked while they stream: every few chunks, and once at the end. Optional settings, under `litellm_params`:

| Setting | Default | What it does |
|---------|---------|--------------|
| `streaming_sampling_rate` | `5` | Check the answer so far every Nth chunk. Each check is a Quilr call. |
| `streaming_end_of_stream_only` | `false` | `true`: check once, after the whole answer has streamed (only 1 Quilr call, but the answer is already delivered). |
| `streaming_buffer_until_moderated` | `false` | `true`: send nothing to the client until the check passes (no live streaming). |
| `streaming_buffer_release_on_scan` | `false` | With buffering: release the held chunks after each passing check. |
| `streaming_transform_mode` | `block_only` | `incremental_diff`: also redact while streaming (OpenAI Chat Completions format only). |
| `stream_holdback_chars` | `64` (with `incremental_diff`) | Characters held back so a value found in them can still be redacted. |

What each setup does:

| Setup | Block | Redact | Quilr calls per answer |
|-------|-------|--------|------------------------|
| Default | Yes: the stream ends with the block message (text already sent stays) | No | Every 5 chunks |
| `streaming_transform_mode: incremental_diff` | Yes | Yes | Every 5 chunks |
| `streaming_end_of_stream_only: true` | Only after the full answer was sent | No | 1 |
| `streaming_buffer_until_moderated: true` | Yes, nothing is sent before the check | No | 1 |

> **Note:** only `incremental_diff` redacts while streaming. In the other setups, if Quilr redacts, LiteLLM drops the redaction and the client gets the original text. Use them to block.

Which setup to use:

- Sensitive values must be redacted: `streaming_transform_mode: incremental_diff` (OpenAI Chat Completions format only)
- Block only, and nothing may reach the client before the check: `streaming_buffer_until_moderated: true`
- Fewest Quilr calls (monitoring): `streaming_end_of_stream_only: true`, or a higher `streaming_sampling_rate`

Example (redaction while streaming):

```yaml
  - guardrail_name: "quilr-output"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrUnifiedGuardrail
      mode: "post_call"
      default_on: true
      streaming_transform_mode: "incremental_diff"
```

#### Logging and tracing (Langfuse, Datadog, ...)

Each check adds a guardrail record to LiteLLM's logs, so it shows in Langfuse and the other LiteLLM logging integrations:

- `guardrail_provider`: `quilr`
- `guardrail_status`: `success`, `guardrail_intervened` (blocked), `guardrail_flagged` (categories found, allowed) or `guardrail_failed_to_respond` (Quilr unreachable)
- `guardrail_response`: Quilr's status, action and categories
- `masked_entity_count`: redacted categories

The record never contains the checked content.

To group checks by conversation and user in Quilr, the guardrail sends:

- `X-Conversation-Id`: the LiteLLM session ID (`metadata.session_id` or `litellm_session_id`)
- `X-User-Email`: the first email among `metadata.trace_user_id`, the request's `user`, the end user ID and the key owner's email
- the session and trace IDs in the request metadata, to match Quilr logs with LiteLLM / Langfuse traces

If Quilr doesn't accept the email (for example, its domain isn't allowed for the app), the check is repeated without it.

#### When Quilr can't be reached

By default, content passes through unchecked when Quilr can't be reached (timeout, connection or API error), and a warning is logged. To block it instead:

```yaml
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrUnifiedGuardrail
      mode: "pre_call"
      quilr_fail_closed: true
```

or set `QUILR_GUARDRAILS_FAIL_CLOSED=true`. Blocked requests get HTTP 503. LiteLLM's `unreachable_fallback` setting is not used: LiteLLM passes it as `fail_closed` even when it isn't configured.

### Optional Filtering

You can optionally limit which requests have guardrails applied using these environment variables:

```bash
# Only apply guardrails to specific models (comma-separated)
APPLY_QUILR_GUARDRAILS_FOR_MODELS=gpt-4,gpt-4o,claude-3-opus

# Only apply guardrails to specific API key names (comma-separated)
APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES=production-key,user-facing-key
```

If neither variable is set, guardrails apply to all requests. If both are set, a request must match both filters (AND logic) for guardrails to be applied.
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
- Blocking, redaction, the model filter and the timeout behave the same as `QuilrGuardrail`.
- Key-name filter: LiteLLM does not pass the key name on every route (e.g. Anthropic `/v1/messages`). On those routes, if `APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES` is set, requests are not checked.
- Requires a LiteLLM version with `apply_guardrail` support (tested with 1.106.2).

### Optional Filtering

You can optionally limit which requests have guardrails applied using these environment variables:

```bash
# Only apply guardrails to specific models (comma-separated)
APPLY_QUILR_GUARDRAILS_FOR_MODELS=gpt-4,gpt-4o,claude-3-opus

# Only apply guardrails to specific API key names (comma-separated)
APPLY_QUILR_GUARDRAILS_FOR_KEY_NAMES=production-key,user-facing-key
```

If neither variable is set, guardrails apply to all requests. If both are set, a request must match both filters (AND logic) for guardrails to be applied.
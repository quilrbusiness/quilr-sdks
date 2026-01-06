## Using Quilr Guardrails with Litellm

### Prerequisites

- Quilr API Key from Quilr guardrails (from either self hosted Quilr guardrails or Quilr hosted)
- Download [quilr_litellm_guardrails.py](https://github.com/quilrbusiness/quilr-sdks/blob/main/python-sdks/litellm_guardrails/quilr_litellm_guardrails.py) and place it in the same directory as your litellm config.yaml file

### Using with litellm

- Add the following to your litellm config.yaml file

```yaml
guardrails:
  - guardrail_name: "quilr-input"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrGuardrail
      mode: "pre_call"

  - guardrail_name: "quilr-output"
    litellm_params:
      guardrail: quilr_litellm_guardrails.QuilrGuardrail
      mode: "post_call"
```

- Set the following environment variables

```bash
QUILR_GUARDRAILS_KEY=sk-quilr-XXXXXXXXX
QUILR_GUARDRAILS_BASE_URL=QUILR GUARDRAILS BASE URL
```
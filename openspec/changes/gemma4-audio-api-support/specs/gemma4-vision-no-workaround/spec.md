## ADDED Requirements

### Requirement: Gemma 4 server starts without multimodal budget workaround
vllm-metal SHALL start an inference server for any `gemma4` model with `is_multimodal_model=True` without requiring `--limit-mm-per-prompt '{"image": 0}'`. The `MultiModalBudget` init in the EngineCore subprocess SHALL compute image token counts without raising `AttributeError`.

#### Scenario: Clean server startup
- **WHEN** `vllm serve google/gemma-4-E4B-it --max-model-len 8192` is run without `--limit-mm-per-prompt`
- **THEN** the server starts successfully and `GET /health` returns HTTP 200 within 5 minutes

#### Scenario: Proxy method resolver handles unregistered attribute
- **WHEN** `MultiModalProcessingInfo.get_max_image_tokens()` is called in the EngineCore subprocess where `Gemma4Processor._get_num_multimodal_tokens` is not registered on the proxy
- **THEN** `_resolve_processor_method()` resolves the method from the concrete class module and returns the correct image token count without raising `AttributeError`

#### Scenario: Compat patch fires in EngineCore
- **WHEN** `MetalModelRunner.__init__` is called in the EngineCore subprocess
- **THEN** `apply_compat_patches()` is called before any multimodal budget computation and the proxy resolver is in place

### Requirement: Vision (image+text) requests succeed without workaround flag
vllm-metal SHALL process `POST /v1/chat/completions` requests containing `"type": "image_url"` content items for Gemma 4 models and return HTTP 200 with a non-empty `choices[0].message.content`.

#### Scenario: Vision request with inline base64 image
- **WHEN** a client sends a chat completion request with a base64-encoded JPEG `"image_url"` content item alongside a text question to a Gemma 4 server started without `--limit-mm-per-prompt`
- **THEN** the response is HTTP 200 and `choices[0].message.content` is a non-empty string

#### Scenario: No regression on text-only requests
- **WHEN** a client sends a text-only chat completion request to a Gemma 4 server
- **THEN** the response is HTTP 200 and `choices[0].message.content` is a non-empty string

#### Scenario: Routing unit test
- **WHEN** the unit test suite is run with model types `["gemma4", "llava", "qwen2_vl", "llama", "qwen3"]`
- **THEN** `gemma4`, `llava`, `qwen2_vl` route through `mlx_vlm_load()` and `llama`, `qwen3` route through `mlx_lm_load()` (5/5 pass)

## 1. Proxy Resolver & Vision Startup Fix

- [x] 1.1 Add `_resolve_processor_method(self, processor)` helper to `MultiModalProcessingInfo` in `vllm/model_executor/models/transformers/multimodal.py`; resolves `_get_num_multimodal_tokens` from concrete class module when proxy attribute lookup fails
- [x] 1.2 Refactor `get_max_image_tokens()` to use `_resolve_processor_method` instead of bare `getattr`
- [x] 1.3 Add `apply_compat_patches()` call to `MetalModelRunner.__init__` so patches fire in the EngineCore subprocess
- [x] 1.4 Add `_patch_gemma4_multimodal_token_budget()` to `vllm_metal/compat.py` mirroring the direct vLLM patch; call from `apply_compat_patches()`
- [x] 1.5 Verify `GET /health` returns HTTP 200 for `google/gemma-4-E4B-it` started without `--limit-mm-per-prompt`
- [x] 1.6 Verify vision (image+text) `POST /v1/chat/completions` returns HTTP 200 with non-empty response

## 2. Audio Modality Budget

- [x] 2.1 Add `_processor_supports_audio(self) → bool` to `MultiModalProcessingInfo`; calls `_get_num_multimodal_tokens(audio_lengths=[16_000])` via resolver, checks `"num_audio_tokens"` in result
- [x] 2.2 Add `get_max_audio_tokens(self) → int`; calls `_get_num_multimodal_tokens(audio_lengths=[30*16_000])` via resolver; fallback returns 1500
- [x] 2.3 Extend `get_supported_mm_limits()` to include `"audio": None` when `_processor_supports_audio()` is True
- [x] 2.4 Extend `get_mm_max_tokens_per_item()` to include `"audio"` key when `"audio" in mm_counts`
- [x] 2.5 Install `librosa`, `soundfile`, `scipy` into vllm-metal venv; confirm `import librosa` succeeds

## 3. Audio Request Routing in TransformersMultiModalForCausalLM

- [ ] 3.1 Extend `MultiModalProcessor.apply()` to handle `"audio"` modality: extract features via `Gemma4Processor.__call__(audios=[...])`, embed via `mlx_vlm_model.embed_audio(features)`, splice into token sequence at `<audio_soft_token>` positions
- [ ] 3.2 Extend `get_mrope_input_positions()` to assign flat-sequence position IDs for audio token spans (no 2D grid encoding); remove the `NotImplementedError("Transformers modeling backend only supports images.")` guard for `gemma4` models
- [ ] 3.3 Add `[audio]` optional extras group to `pyproject.toml`: `librosa`, `soundfile`, `scipy`
- [ ] 3.4 Update `INSTALL.md` with an "Audio Support" section describing `pip install vllm-metal[audio]`

## 4. Integration Tests

- [ ] 4.1 Text-only non-regression: `POST /v1/chat/completions` returns HTTP 200 with non-empty response (Gemma 4 server, no multimodal flags)
- [ ] 4.2 Vision non-regression: `POST /v1/chat/completions` with base64 JPEG returns HTTP 200 with non-empty response
- [ ] 4.3 Audio: `POST /v1/chat/completions` with base64 WAV `"audio_url"` returns HTTP 200 with non-empty `choices[0].message.content`
- [ ] 4.4 Confirm unit routing tests still pass (5/5: gemma4/llava/qwen2_vl → mlx-vlm; llama/qwen3 → mlx-lm)

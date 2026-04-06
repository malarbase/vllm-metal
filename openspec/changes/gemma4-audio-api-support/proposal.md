## Why

vllm-metal's Gemma 4 vision path was blocked at startup by a `Gemma4Processor._get_num_multimodal_tokens` `AttributeError` in vLLM's EngineCore subprocess, and Gemma 4 audio input is entirely absent from the vLLM OpenAI-compatible API — despite mlx-vlm's Conformer audio encoder being fully implemented in `mlx_vlm/models/gemma4/audio.py`.

This change fixes both gaps: the startup crash that prevented vision from working without the `--limit-mm-per-prompt '{"image": 0}'` crutch, and the missing audio API surface that currently returns `NotImplementedError` when a client sends an audio clip.

## What Changes

- **Fix `Gemma4Processor` proxy-object `AttributeError`** in vLLM 0.17.1's EngineCore subprocess: the processor is delivered via IPC as a proxy object whose `_get_num_multimodal_tokens` attribute is not registered. `MultiModalProcessingInfo.get_max_image_tokens` is patched to resolve the method from the concrete class module when the instance lookup fails, eliminating the need for `--limit-mm-per-prompt '{"image": 0}'`.
- **Add compat patch hook in `MetalModelRunner.__init__`**: vllm-metal's `apply_compat_patches()` is called early in `MetalModelRunner.__init__` so it fires in the EngineCore subprocess (where the platform plugin's `register()` hook does not run).
- **Add `_patch_gemma4_multimodal_token_budget`** to `vllm_metal/compat.py`: a belt-and-suspenders patch that replaces `MultiModalProcessingInfo.get_max_image_tokens` with a proxy-resilient version in both the APIServer and any subprocess where the platform plugin fires.
- **Expose audio modality in `MultiModalProcessingInfo`**: extend `get_supported_mm_limits()` and `get_mm_max_tokens_per_item()` in the patched vLLM file to include `"audio"` when `Gemma4Processor._get_num_multimodal_tokens` reports audio token support. Add `get_max_audio_tokens()` that computes the budget for a 30-second clip at 16 kHz.
- **Add audio routing in `TransformersMultiModalForCausalLM`**: implement audio input processing (feature extraction via `Gemma4Processor`, embedding via mlx-vlm's `embed_audio`) in the `MultiModalProcessor` and hook it into `get_mrope_input_positions` so audio tokens reach the MLX forward pass instead of raising `NotImplementedError`.

## Capabilities

### New Capabilities

- `gemma4-vision-no-workaround`: Gemma 4 vision (image+text) requests work without the `--limit-mm-per-prompt '{"image": 0}'` startup flag; the server now starts cleanly with the full multimodal budget computed.
- `gemma4-audio-api`: vLLM's OpenAI-compatible `/v1/chat/completions` endpoint accepts `"type": "audio_url"` content items for Gemma 4 models, resamples the audio to 16 kHz, extracts features via `Gemma4Processor`, embeds via mlx-vlm's `embed_audio`, and returns a non-empty `choices[0].message.content`.

### Modified Capabilities

_(none — existing text and vision request paths are unchanged)_

## Impact

- **`vllm_metal/v1/model_runner.py`**: `MetalModelRunner.__init__` gains an `apply_compat_patches()` call.
- **`vllm_metal/compat.py`**: new `_patch_gemma4_multimodal_token_budget()` function; called from `apply_compat_patches()`.
- **`vllm/model_executor/models/transformers/multimodal.py`** (installed vLLM package): patched locally (or via a vllm-metal–managed monkey-patch) to add `_resolve_processor_method()`, extended `get_supported_mm_limits()` / `get_mm_max_tokens_per_item()` / `get_max_audio_tokens()`, and audio handling in `MultiModalProcessor`. This patch should be upstreamed to vLLM once validated.
- **`pyproject.toml`** (vllm-metal): add `librosa`, `soundfile`, `scipy` to optional `audio` extras (mirrors `vllm[audio]`).
- **No API surface changes** — the `/v1/chat/completions` endpoint contract is unchanged; this change enables previously-rejected request types to succeed.
- **Tested configuration**: `google/gemma-4-E4B-it` (5B) on M4 Max 128 GB; generalises to `gemma-4-31B-it` and `gemma-4-2B-it`.

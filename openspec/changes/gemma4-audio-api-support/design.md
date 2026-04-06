## Context

vLLM 0.17.1 uses a `spawn`-based multiprocessing model: the `APIServer` runs in the main process, while inference work (scheduling, KV cache, model execution) runs in an `EngineCore` subprocess spawned fresh. The platform plugin's `register()` hook — and any monkey-patches installed there — only fires in the main process. When `EngineCore` initialises `MetalModelRunner`, patches applied earlier in `APIServer` are absent.

This surfaces in two concrete failures:

1. **Vision startup crash**: `MultiModalProcessingInfo.get_max_image_tokens()` calls `processor._get_num_multimodal_tokens(image_sizes=...)` during `MultiModalBudget` init in `EngineCore`. The `Gemma4Processor` arrives via IPC as a `SyncManager` proxy whose `_get_num_multimodal_tokens` attribute is not registered with the proxy, so the lookup raises `AttributeError`. The workaround (`--limit-mm-per-prompt '{"image": 0}'`) suppresses the budget check at the cost of disabling vision entirely.

2. **Audio returns 400 / `NotImplementedError`**: `TransformersMultiModalForCausalLM` in vLLM 0.17.1 only handles `"image"` in `get_mrope_input_positions` and `MultiModalProcessingInfo.get_supported_mm_limits`. Any `"audio_url"` content item is rejected at the budget stage (`"At most 0 audio(s) may be provided"`) or raises `NotImplementedError("Transformers modeling backend only supports images.")` deep in the forward pass routing. The audio encoder in mlx-vlm (`mlx_vlm/models/gemma4/audio.py`) is fully implemented and wired into the MLX forward pass — the gap is entirely in the vLLM API layer.

**Prior art from `vllm-metal-gemma4-support`:** The routing fix and mlx-vlm dependency wiring are already shipped. The compat patch infrastructure (`vllm_metal/compat.py`, `apply_compat_patches()`) is in place. This change builds on those foundations.

## Goals / Non-Goals

**Goals:**
- Gemma 4 vision works without `--limit-mm-per-prompt '{"image": 0}'`
- vLLM's OpenAI-compatible API accepts `"audio_url"` inputs for Gemma 4 and returns a non-empty response
- All fixes are idempotent and do not regress existing text or vision paths
- Core logic landed in the vllm-metal plugin where possible; direct vLLM file patches are documented for upstreaming

**Non-Goals:**
- Supporting audio for non-Gemma 4 Transformers-backend models (not tested, not scoped)
- Upstreaming audio patches to vLLM core (tracked as a follow-on)
- Video input (mlx-vlm has no Gemma 4 video tower)

## Decisions

### D1: Fix the proxy `AttributeError` by resolving the unbound method from the concrete class module

**Choice:** Extend `MultiModalProcessingInfo` with a `_resolve_processor_method(processor)` helper. When `getattr(processor, "_get_num_multimodal_tokens")` fails (proxy gap), it imports the processor's declared module and calls `getattr(class, method_name)` to get the unbound function, then wraps it in a `MethodType` bound to the proxy instance. Calls to `get_max_image_tokens()` and `get_max_audio_tokens()` use this resolver instead of bare `getattr`.

**Rationale:** The proxy object faithfully delegates registered attributes and methods; the attribute simply isn't registered. Resolving via the module bypasses the proxy for method dispatch while still passing the proxy as `self`, which is safe because all the method accesses inside `_get_num_multimodal_tokens` are on `self.image_processor` / `self.tokenizer` — attributes that are registered on the proxy.

**Alternative considered:** Register `_get_num_multimodal_tokens` in the `SyncManager` proxy class. Rejected because it requires modifying vLLM's multiprocessing setup and would need to be re-applied on every vLLM upgrade. The resolver approach is a targeted, upgrade-resilient shim.

**Alternative considered:** Materialise the processor in `APIServer` before it reaches `EngineCore` (pickle the concrete object). Rejected because the proxy architecture is intentional in vLLM — it avoids serialising the full `PreTrainedProcessor` over the IPC pipe on every request.

### D2: Land the fix in a direct patch to `vllm/model_executor/models/transformers/multimodal.py`, surfaced via `vllm_metal/compat.py`

**Choice:** The patched file is inside the installed vLLM package. vllm-metal wraps the same logic in `_patch_gemma4_multimodal_token_budget()` in `compat.py`. Both coexist: the direct patch ensures correctness even if the compat hook fires late; the compat hook documents the intent and is the canonical path for upstreaming.

`MetalModelRunner.__init__` calls `apply_compat_patches()` explicitly, which runs in `EngineCore`. The platform plugin's `register()` call in `APIServer` is a second activation site — idempotency (`_APPLIED` guard) prevents double-patching.

**Rationale:** The direct patch is pragmatic: it fixes the bug unconditionally today with no process-timing dependency. The compat wrapper is the long-term mechanism that survives vLLM version bumps (re-applies the patch) and communicates the issue to maintainers. Both are cheap to maintain together.

### D3: Expose `"audio"` modality by extending `MultiModalProcessingInfo`, not by subclassing

**Choice:** Add three methods to `MultiModalProcessingInfo` in the patched vLLM file:
- `get_max_audio_tokens(self) → int` — calls `_get_num_multimodal_tokens(audio_lengths=[30*16_000])` via the resolver; falls back to 1500 tokens if the call fails.
- `get_supported_mm_limits(self) → dict` — returns `{"image": None, "audio": None}` when `_processor_supports_audio()` is True; `{"image": None}` otherwise.
- `get_mm_max_tokens_per_item(self, seq_len, mm_counts) → dict` — includes `"audio"` key when `"audio" in mm_counts`.

`_processor_supports_audio()` calls `_get_num_multimodal_tokens(audio_lengths=[16_000])` and checks for `"num_audio_tokens"` in the result.

**Rationale:** Subclassing would require changing `TransformersMultiModalForCausalLM`'s instantiation path. Extending the existing class via the direct patch keeps the diff minimal and maximises forward compatibility (a future vLLM release that adds proper audio support can land its own `get_max_audio_tokens` and the override will be superseded cleanly).

### D4: Implement audio input routing inside `TransformersMultiModalForCausalLM`

**Choice:** Extend `MultiModalProcessor.apply()` and `get_mrope_input_positions()` to handle `"audio"` modality items:

1. In `apply()`: when the input contains audio items, run `Gemma4Processor.__call__(audios=[...])` to extract features, then call `mlx_vlm_model.embed_audio(features)` to get embeddings. Merge audio embeddings into the input sequence at the `<audio_soft_token>` positions using the same splice logic used for image embeddings.
2. In `get_mrope_input_positions()`: map audio token spans in the flattened token sequence to the correct MRoPE position IDs (audio tokens use the same flat-sequence position scheme as image tokens in Gemma 4 — no 2D grid, unlike image patches in Qwen-VL).

**Rationale:** mlx-vlm's `embed_audio` is a direct tensor→embedding call; no additional model changes are needed. The Gemma 4 audio token count is deterministic given the audio length (output of `_get_num_multimodal_tokens`), so position ID generation is straightforward.

**Risk:** The `apply()` and `get_mrope_input_positions()` methods are large and entangled with image logic. Audio changes must be additive (guarded by `if modality == "audio"`) to avoid regressions. Integration tests for text and vision must be re-run after the audio wiring is added.

### D5: Add `librosa`, `soundfile`, `scipy` to `pyproject.toml` as `[audio]` optional extras

**Choice:** Mirror `vllm[audio]`'s dependency set. Document in `INSTALL.md` that audio requires `pip install vllm-metal[audio]`.

**Rationale:** These are large packages (librosa alone is ~3 MB installed). Keeping them optional avoids bloating text-only deployments. The `[audio]` extra mirrors the vLLM convention, so users upgrading from `vllm[audio]` need to add one extra flag.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Proxy resolver uses module import — the processor's `__module__` attribute may not be registered on the proxy | Checked: `type(proxy).__module__` and `type(proxy).__name__` are class-level attributes available without proxy dispatch. |
| `_processor_supports_audio()` calls `_get_num_multimodal_tokens` at startup — non-Gemma 4 processors may return unexpected shapes | Wrapped in `try/except Exception`; returns `False` on any failure. |
| Audio routing in `apply()` and `get_mrope_input_positions()` may collide with a future upstream vLLM audio implementation | Guarded by `if modality == "audio"` blocks; upstream additions will override in a future vLLM upgrade. |
| vLLM 0.17.1 is a pinned dev dependency — direct file patches are fragile on version bump | All patched methods are documented inline with `# vllm-metal-patch` comments; a grep on upgrade will surface them. |
| Audio features extraction via `Gemma4Processor.__call__` is CPU-bound and synchronous | Acceptable for initial implementation; async extraction is a follow-on optimisation. |

## Migration Plan

1. Add `_resolve_processor_method`, `get_max_audio_tokens`, `get_supported_mm_limits`, `get_mm_max_tokens_per_item` to `MultiModalProcessingInfo` in the direct vLLM file patch
2. Update `_patch_gemma4_multimodal_token_budget()` in `vllm_metal/compat.py` to match the new method set
3. Extend `TransformersMultiModalForCausalLM.apply()` and `get_mrope_input_positions()` for audio routing
4. Add `[audio]` extras to `pyproject.toml`; update `INSTALL.md`
5. Run integration tests: text (non-regression), vision (non-regression), audio (new)

**Rollback:** Remove `"audio"` from `get_supported_mm_limits()` to collapse back to images-only. No data migration or API changes required.

## Open Questions

1. **MRoPE position IDs for audio** — Gemma 4 uses flat positions for audio tokens (confirmed from `mlx_vlm` source). Verify with a forward-pass trace that position IDs do not need a 2D or special encoding.
2. **Upstream vLLM PR** — Should the `MultiModalProcessingInfo` audio extensions be proposed to vLLM core now, or wait until the `TransformersMultiModalForCausalLM` audio routing is stable? Current plan: stabilise locally first, then open PR.
3. **`vllm-metal-gemma4-support` compat patch fate** — Now that the direct vLLM file patch is the primary fix, should `_patch_gemma4_multimodal_token_budget()` in `compat.py` be simplified to a no-op that logs a warning if the vLLM file is not already patched? Deferred: keep the compat patch as a belt-and-suspenders fix until audio is fully shipped.

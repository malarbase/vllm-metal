# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for vLLM + transformers version mismatches.

Applied once at platform registration time.  Each patch is guarded by
try/except so it degrades silently if the target module changes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False
_MM_BUDGET_PATCHED = False


def apply_compat_patches() -> None:
    """Apply all known compatibility patches (idempotent).

    This may be called from two entry points:
    - ``_register_compat`` (vllm.general_plugins) — fired early in every
      process via ``load_general_plugins()`` before the Scheduler is created.
    - ``_register`` (vllm.platform_plugins) — fired lazily when
      ``current_platform`` is first resolved.

    The outer ``_APPLIED`` guard deduplicates the cheap patches, but the
    multimodal patch has its own ``_MM_BUDGET_PATCHED`` sentinel because it
    may fail silently on the first call (e.g. the platform plugin fires
    before ``vllm.model_executor`` is fully initialised) while still
    marking ``_APPLIED=True``.  The general plugin always retries it.
    """
    global _APPLIED  # noqa: PLW0603
    if not _APPLIED:
        _APPLIED = True
        _patch_qwen35_rope_validation()
        _patch_gemma4_rope_scaling()
    # Always attempt the multimodal patch — it has its own idempotency guard.
    _patch_gemma4_multimodal_token_budget()


def _patch_qwen35_rope_validation() -> None:
    """Fix vLLM 0.17.1 Qwen3.5 config vs transformers >=5.4 rope validation.

    vLLM's ``Qwen3_5TextConfig.__init__`` hardcodes
    ``kwargs["ignore_keys_at_rope_validation"] = [...]`` (a list), but
    transformers 5.4+ does ``received_keys -= ignore_keys`` which requires
    a set.

    Upstream: vllm-project/vllm#34604 fixed this but was reverted in #34610.
    Remove this patch when vllm-metal upgrades to a vLLM version with the fix.
    """
    from importlib.util import find_spec

    if find_spec("vllm.transformers_utils.configs.qwen3_5") is None:
        return

    try:
        from transformers.modeling_rope_utils import RopeConfigBase

        rope_config_base = RopeConfigBase
    except ImportError:
        rope_config_base = None

    if rope_config_base is None:
        # Try the direct path
        try:
            import transformers.modeling_rope_utils as _rope

            _orig_check = _rope._check_received_keys

            def _safe_check(
                rope_type,
                received_keys,
                required_keys,
                optional_keys=None,
                ignore_keys=None,
            ):
                if ignore_keys is not None and isinstance(ignore_keys, list):
                    ignore_keys = set(ignore_keys)
                return _orig_check(
                    rope_type, received_keys, required_keys, optional_keys, ignore_keys
                )

            _rope._check_received_keys = _safe_check
            logger.debug("Patched _check_received_keys for rope validation compat")
            return
        except (ImportError, AttributeError):
            pass

    # Fallback: patch the static method on PreTrainedConfig if available
    try:
        from transformers import PreTrainedConfig

        if hasattr(PreTrainedConfig, "_check_received_keys"):
            _orig_check = PreTrainedConfig._check_received_keys

            @staticmethod
            def _safe_check(
                rope_type,
                received_keys,
                required_keys,
                optional_keys=None,
                ignore_keys=None,
            ):
                if ignore_keys is not None and isinstance(ignore_keys, list):
                    ignore_keys = set(ignore_keys)
                return _orig_check(
                    rope_type, received_keys, required_keys, optional_keys, ignore_keys
                )

            PreTrainedConfig._check_received_keys = _safe_check
            logger.debug(
                "Patched PreTrainedConfig._check_received_keys for rope compat"
            )
    except (ImportError, AttributeError):
        pass


def _patch_gemma4_rope_scaling() -> None:
    """Fix vLLM rope_scaling validation for Gemma 4.

    Transformers maps Gemma 4's ``rope_parameters`` field onto the
    ``rope_scaling`` attribute of ``Gemma4TextConfig``.  The resulting dict
    uses a *nested* format::

        {"full_attention": {"rope_type": "proportional", ...},
         "sliding_attention": {"rope_type": "default", ...}}

    vLLM's ``patch_rope_scaling_dict`` expects a *flat* dict with a top-level
    ``rope_type`` key and raises ``ValueError`` when it is absent.  Since vLLM
    has no way to run the Gemma 4 rope calculation itself (it delegates to MLX),
    we can safely skip the flat-format validation for nested dicts.

    Remove this patch when vLLM adds native Gemma 4 rope_scaling support.
    """
    try:
        import vllm.transformers_utils.config as _cfg_module

        _orig_patch = _cfg_module.patch_rope_scaling_dict

        def _safe_patch_rope_scaling_dict(rope_scaling: dict) -> None:  # type: ignore[override]
            # Nested format (Gemma 4) — values are sub-dicts, not scalars.
            if any(isinstance(v, dict) for v in rope_scaling.values()):
                return
            return _orig_patch(rope_scaling)

        _cfg_module.patch_rope_scaling_dict = _safe_patch_rope_scaling_dict
        logger.debug(
            "Patched patch_rope_scaling_dict for Gemma 4 rope_parameters compat"
        )
    except (ImportError, AttributeError):
        pass


def _patch_gemma4_multimodal_token_budget() -> None:  # noqa: PLR0912
    """Fix vLLM 0.17.1 EngineCore subprocess failing to call
    ``Gemma4Processor._get_num_multimodal_tokens`` during MultiModalBudget init.

    Root cause: in vLLM's spawn-based EngineCore subprocess the processor
    object is delivered via IPC (shared memory / pickle).  The reconstructed
    object's type reports ``Gemma4Processor`` but attribute lookup for
    ``_get_num_multimodal_tokens`` fails, most likely because the proxy
    wrapper only exposes a predefined set of proxied attributes and this
    method—added in transformers 5.5.0—was not in the registered set when
    the proxy class was generated.

    Fix: wrap ``MultiModalProcessingInfo.get_max_image_tokens`` so that when
    the attribute is unreachable on the instance we resolve it directly from
    the concrete class in the transformers module, bypassing the proxy layer.

    Has its own ``_MM_BUDGET_PATCHED`` sentinel (separate from the outer
    ``_APPLIED`` guard) because ``apply_compat_patches`` may be called early
    via the platform plugin before ``vllm.model_executor`` is importable; the
    import would return early but ``_APPLIED`` would already be True, blocking
    a later retry.  This patch is always retried until it succeeds.

    Remove this patch when vllm-metal upgrades to a vLLM version that has
    been tested against transformers 5.5.0 with Gemma 4.
    """
    global _MM_BUDGET_PATCHED  # noqa: PLW0603
    if _MM_BUDGET_PATCHED:
        return

    try:
        from vllm.model_executor.models.transformers import multimodal as _mm
    except ImportError:
        return

    try:
        _orig = _mm.MultiModalProcessingInfo.get_max_image_tokens

        def _patched_get_max_image_tokens(self) -> int:  # type: ignore[override]
            width, height = self.get_max_image_size()
            processor = self.get_hf_processor()
            multimodal_config = self.ctx.model_config.multimodal_config
            mm_processor_kwargs = (multimodal_config.mm_processor_kwargs or {})

            method = getattr(processor, "_get_num_multimodal_tokens", None)
            if method is None:
                # Proxy object: resolve unbound function from the concrete class.
                import importlib

                proc_cls_name = type(processor).__name__
                proc_module = importlib.import_module(type(processor).__module__)
                proc_cls = getattr(proc_module, proc_cls_name, None)
                if proc_cls is not None:
                    unbound = getattr(proc_cls, "_get_num_multimodal_tokens", None)
                    if unbound is not None:
                        import types
                        method = types.MethodType(unbound, processor)

            if method is None:
                logger.warning(
                    "Could not resolve _get_num_multimodal_tokens for %s; "
                    "falling back to conservative default of 256 image tokens",
                    type(processor).__name__,
                )
                return 256

            mm_tokens = method(
                image_sizes=([height, width],), **mm_processor_kwargs
            )
            return mm_tokens["num_image_tokens"][0]

        _mm.MultiModalProcessingInfo.get_max_image_tokens = _patched_get_max_image_tokens
        _MM_BUDGET_PATCHED = True
        logger.debug(
            "Patched MultiModalProcessingInfo.get_max_image_tokens "
            "for Gemma4Processor proxy compat"
        )
    except (ImportError, AttributeError):
        pass

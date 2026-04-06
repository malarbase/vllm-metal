## ADDED Requirements

### Requirement: Audio modality is included in the multimodal budget for Gemma 4
vllm-metal SHALL configure vLLM's `MultiModalProcessingInfo` so that `"audio"` is listed as a supported modality for Gemma 4 models, with a token budget derived from `Gemma4Processor._get_num_multimodal_tokens(audio_lengths=[...])`.

#### Scenario: Audio modality listed in supported limits
- **WHEN** `MultiModalProcessingInfo.get_supported_mm_limits()` is called for a Gemma 4 model
- **THEN** the returned dict contains both `"image"` and `"audio"` keys

#### Scenario: Audio token budget is computed correctly
- **WHEN** `MultiModalProcessingInfo.get_max_audio_tokens()` is called for a Gemma 4 model
- **THEN** it returns an integer > 0 representing the token count for 30 seconds of 16 kHz audio (approx. 1500 tokens if the processor call fails)

#### Scenario: Non-Gemma processor falls back to images-only
- **WHEN** `get_supported_mm_limits()` is called for a model whose processor does not implement `_get_num_multimodal_tokens` with `audio_lengths` support
- **THEN** the returned dict contains only `"image"` and no `"audio"` key

### Requirement: Audio input is accepted and processed by the OpenAI-compatible API
vllm-metal SHALL process `POST /v1/chat/completions` requests containing `"type": "audio_url"` content items for Gemma 4 models and return HTTP 200 with a non-empty `choices[0].message.content`.

#### Scenario: Audio request with WAV base64 input
- **WHEN** a client sends a chat completion request with a base64-encoded WAV `"audio_url"` content item alongside a text question to a Gemma 4 server with `[audio]` extras installed
- **THEN** the response is HTTP 200 and `choices[0].message.content` is a non-empty string

#### Scenario: Audio features are extracted and embedded correctly
- **WHEN** an audio input is received by `TransformersMultiModalForCausalLM`
- **THEN** `Gemma4Processor` extracts mel-spectrogram features, `mlx_vlm_model.embed_audio` converts them to token embeddings, and the embeddings are spliced into the token sequence at `<audio_soft_token>` positions

#### Scenario: MRoPE positions are assigned for audio tokens
- **WHEN** `get_mrope_input_positions()` processes a sequence containing audio tokens
- **THEN** each audio token is assigned a valid flat-sequence position ID (no 2D grid encoding) without raising `NotImplementedError`

#### Scenario: Audio optional extras gate the feature
- **WHEN** `librosa`, `soundfile`, and `scipy` are not installed and a client sends an audio request
- **THEN** the server returns a clear error indicating that audio extras are required (HTTP 500 with `"install vllm-metal[audio]"` message)

### Requirement: Audio extras are documented and installable
vllm-metal SHALL document the `[audio]` optional dependency group in `pyproject.toml` and `INSTALL.md` so users can enable audio support with a single pip command.

#### Scenario: Audio extras install successfully
- **WHEN** `pip install vllm-metal[audio]` is run
- **THEN** `librosa`, `soundfile`, and `scipy` are installed and `import librosa` succeeds

#### Scenario: INSTALL.md includes audio setup section
- **WHEN** a user reads `INSTALL.md`
- **THEN** there is a section explaining that audio input requires `pip install vllm-metal[audio]`

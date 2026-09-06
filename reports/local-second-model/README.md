# Second local model readiness

The installed Nemotron 3 Nano 4B Q4_K_M model is loaded as `factor-nemotron4b`. Use Factor's `LocalChatClient` with `transport="lmstudio-native"` and the existing loopback server at port 1234.

The task-specific load uses a 4,096-token context, full GPU offload and a 3,600-second idle lifetime. LM Studio reported 16.94 seconds to load and 2.64 GiB loaded. No other model was unloaded, no model weights were downloaded, and shared source files were not edited.

One neutral request asked for `{"ok":true}`. The final message returned that object in 0.456 seconds, with six output tokens and zero reasoning tokens. It used `reasoning: off`, temperature zero and `store: false`. The request and response are in `neutral_probe.json`. No motion evaluation cases were inspected or sent during this preparation.

The complete file identity, SHA-256, matching repository file commit and load settings are in `model_manifest.json`. The local 2,837,072,896-byte file matches the repository's published hash. [Quantized-file commit](https://huggingface.co/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/commit/2a5be5ca635ed930b4629b3f3600a7c8a1c3c84b)

NVIDIA's model card identifies the base as a text model using a hybrid architecture and governed by the NVIDIA Nemotron Open Model License. [Official NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)

This readiness check establishes a working serving path. The root task will compare it with Qwen on the same frozen motion protocol. A neutral JSON response establishes no scientific performance or model superiority.

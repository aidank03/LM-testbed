# Visual expansion verification

Added 27 visual labs to notebooks 05–14, bringing the course to **54 static figures and two embedded animations**. Each new lab includes a reading prompt and executable figure code. The animations cover a simulated shot and autoregressive token generation, with static fallback figures.

All ten notebooks executed successfully in fresh Jupyter kernels after installation. Their original numerical metrics match the preceding course run exactly. Original code and prose cells were retained; earlier notebooks 01–04 are byte-for-byte unchanged. Pre-edit course versions are backed up under `runs/world-model/visuals-v1-originals/`.

The visual insertion helper is idempotent and the authoring script includes it, so deliberate notebook regeneration retains the visual labs. The insertion helper checks each heading anchor before changing a notebook. No new dependency, model download or API call was introduced.

[Verification data](verification.json) · [Current metrics and source hashes](metrics.json) · [Visual learning guide](../../docs/VISUAL_LEARNING_GUIDE.md)

New static figures are available as individual PNGs in this directory. Three contact sheets collect them for review. The initial contact-sheet review covered all 27 additions; a final check covered the revised teacher-forcing diagram. Animation generation executed successfully and embedded frames and controls in the notebook HTML outputs; cross-viewer JavaScript playback is dependent on notebook trust and renderer support.

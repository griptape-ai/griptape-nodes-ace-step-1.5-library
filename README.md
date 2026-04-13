# Griptape Nodes ACE-Step 1.5 Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for music generation using [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).

## Overview

This library exposes ACE-Step 1.5, an open-source music generation foundation model, as Griptape Nodes. Generate full musical compositions (10 seconds to 10 minutes) from a text prompt and optional lyrics, or produce a style-transfer cover of a reference audio file. The model supports 1000+ instruments and styles, 50+ vocal languages, and runs on CUDA (Windows/Linux) and MPS/MLX (Apple Silicon). Audio is output at 48kHz stereo FLAC quality.

## Requirements

- **GPU**: CUDA (NVIDIA) or MPS (Apple Silicon) required
- **Griptape Nodes Engine**: Version 0.77.5 or later

## Nodes

### Text to Music

Generate a musical composition from a text caption and optional lyrics.

**Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `caption` | str | *(required)* | Text prompt describing the desired music style, genre, and mood (max 512 characters) |
| `lyrics` | str | `[Instrumental]` | Song lyrics with structure tags (e.g. `[Verse 1]`, `[Chorus]`). Use `[Instrumental]` for no vocals |
| `duration` | float | `-1.0` | Target audio length in seconds (10-600). Use -1 for automatic duration |
| `dit_model` | HuggingFace model | `ACE-Step/Ace-Step1.5` | DiT model variant to use for generation |
| `inference_steps` | int | `8` | Number of diffusion steps. Use 8 for turbo models, 32-100 for base/sft models |
| `guidance_scale` | float | `7.0` | Classifier-free guidance strength (only effective for non-turbo models) |
| `seed` | int | `-1` | Random seed for reproducibility. -1 uses a random seed |
| `bpm` | int | `0` | Target beats per minute (30-300). Leave 0 for automatic detection |
| `vocal_language` | str | `unknown` | Vocal language code (e.g. `en`, `zh`, `ja`). Use `unknown` for auto-detection |
| `enable_lm` | bool | `False` | Enable 5Hz Language Model for Chain-of-Thought metadata and audio code generation |
| `lm_model` | HuggingFace model | `ACE-Step/Ace-Step1.5` | Language Model variant to load when `enable_lm` is enabled |
| `device` | str | `auto` | Compute device: `auto`, `cuda`, `mps`, or `cpu` |

**Output:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | AudioUrlArtifact | Generated audio (48kHz stereo FLAC) |

### Audio Cover

Generate a style-transfer cover of a reference audio file, optionally guided by a new caption and lyrics.

**Inputs:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_audio` | AudioArtifact / AudioUrlArtifact | *(required)* | Source audio to create a cover from |
| `caption` | str | *(empty)* | Text description for the new style. Leave empty to preserve the original style |
| `lyrics` | str | *(empty)* | New lyrics for the cover. Leave empty to preserve original lyrics |
| `audio_cover_strength` | float | `1.0` | How closely to follow the reference audio (0.0 = creative freedom, 1.0 = close to original) |
| `dit_model` | HuggingFace model | `ACE-Step/Ace-Step1.5` | DiT model variant to use for generation |
| `inference_steps` | int | `8` | Number of diffusion steps |
| `seed` | int | `-1` | Random seed for reproducibility. -1 uses a random seed |
| `device` | str | `auto` | Compute device: `auto`, `cuda`, `mps`, or `cpu` |

**Output:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | AudioUrlArtifact | Generated cover audio (48kHz stereo FLAC) |

## Available Models

Models are downloaded automatically on first use and cached for subsequent runs.

### DiT Models (required for all nodes)

`ACE-Step/Ace-Step1.5` must be downloaded first -- it contains the shared VAE and text encoder required by all variants, and also bundles the default turbo DiT and 1.7B LM.

| Model | Description |
|-------|-------------|
| `ACE-Step/Ace-Step1.5` | Main repo: bundles the turbo DiT (2B) + 1.7B LM. **Download this first.** |
| `ACE-Step/acestep-v15-sft` | 2B DiT, supervised fine-tuning variant, higher quality (32-100 steps) |
| `ACE-Step/acestep-v15-base` | 2B DiT, base pre-training variant |
| `ACE-Step/acestep-v15-xl-base` | 4B XL DiT, base variant, requires 12GB+ VRAM |
| `ACE-Step/acestep-v15-xl-sft` | 4B XL DiT, SFT variant, best quality |
| `ACE-Step/acestep-v15-xl-turbo` | 4B XL DiT, turbo variant |

### Language Models (optional, used when `enable_lm` is enabled)

| Model | Description |
|-------|-------------|
| `ACE-Step/Ace-Step1.5` | Bundles the 1.7B LM (same download as the main DiT repo) |
| `ACE-Step/acestep-5Hz-lm-0.6B` | 0.6B LM, fastest, requires ~2GB VRAM |
| `ACE-Step/acestep-5Hz-lm-4B` | 4B LM, highest quality CoT, requires ~16GB+ total VRAM |

## Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- A CUDA-capable NVIDIA GPU or Apple Silicon Mac

### Install the Library

1. **Clone the repository** to your Griptape Nodes workspace directory:

   ```bash
   cd `gtn config show workspace_directory`
   git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-ace-step-1.5-library.git
   ```

2. **Add the library** in the Griptape Nodes Editor:

   - Open the Settings menu and navigate to the *Libraries* settings
   - Click on *+ Add Library* at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-ace-step-1.5-library/griptape_nodes_ace_step_1_5_library/griptape-nodes-library.json
     ```
   - You can check your workspace directory with `gtn config show workspace_directory`
   - Close the Settings Panel
   - Click on *Refresh Libraries*

3. **Verify installation** by checking that the nodes appear in the node palette under the "Music Generation" category.

## Usage

### Text to Music

1. Add a **Text to Music** node to your workflow
2. Set the `caption` to a description of the music you want (e.g. `"upbeat jazz piano trio, lively swing rhythm"`)
3. Optionally set `lyrics` with structure tags, or leave the default `[Instrumental]`
4. Select a DiT model from the `dit_model` dropdown (requires `ACE-Step/Ace-Step1.5` to be downloaded first)
5. Connect the `audio` output to a display node or further processing

### Audio Cover

1. Add an **Audio Cover** node to your workflow
2. Connect a source audio file to the `reference_audio` input
3. Optionally provide a `caption` to steer the style of the cover
4. Adjust `audio_cover_strength` to control how closely the output follows the original
5. Connect the `audio` output to your next node

## Troubleshooting

### Library Not Loading

- Ensure the git submodule is initialized. If you cloned without `--recurse-submodules`, run:
  ```bash
  git submodule update --init --recursive
  ```

### CUDA / MPS Not Available

- Verify your GPU drivers are up to date
- For NVIDIA GPUs, ensure CUDA is properly installed
- For Apple Silicon, ensure you are running macOS 12.3 or later

### Out of Memory Errors

- Try using a smaller DiT variant (e.g. `acestep-v15-turbo` instead of an XL model)
- Disable `enable_lm` or switch to the 0.6B LM model
- Close other GPU-intensive applications

## Additional Resources

- [ACE-Step 1.5 GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is provided under the Apache License 2.0. The bundled ACE-Step 1.5 submodule is subject to its own license: MIT.

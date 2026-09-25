import logging
import os
import sys
import tempfile
import uuid
from typing import Any

from griptape.artifacts import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

logger = logging.getLogger("ace_step_1_5_library")

# Real HuggingFace repo IDs for DiT models.
# "ACE-Step/Ace-Step1.5" is the main repo and bundles the default turbo DiT.
# All others are separate repos downloadable via huggingface-cli.
DIT_MODEL_REPO_IDS = [
    "ACE-Step/Ace-Step1.5",
    "ACE-Step/acestep-v15-sft",
    "ACE-Step/acestep-v15-base",
    "ACE-Step/acestep-v15-xl-base",
    "ACE-Step/acestep-v15-xl-sft",
    "ACE-Step/acestep-v15-xl-turbo",
]

# Real HuggingFace repo IDs for LM models.
# "ACE-Step/Ace-Step1.5" also bundles the default 1.7B LM.
LM_MODEL_REPO_IDS = [
    "ACE-Step/Ace-Step1.5",
    "ACE-Step/acestep-5Hz-lm-0.6B",
    "ACE-Step/acestep-5Hz-lm-4B",
]

# Maps the main repo ID to the DiT config_path bundled inside it.
# All other repos: config_path = repo_id.split("/")[-1].
_MAIN_REPO_DIT_CONFIG = "acestep-v15-turbo"

# Maps the main repo ID to the LM config_path bundled inside it.
_MAIN_REPO_LM_CONFIG = "acestep-5Hz-lm-1.7B"

MAIN_REPO_ID = "ACE-Step/Ace-Step1.5"

DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]


def _repo_to_dit_config(repo_id: str) -> str:
    """Map a HuggingFace repo ID to the ACE-Step DiT config_path string."""
    if repo_id == MAIN_REPO_ID:
        return _MAIN_REPO_DIT_CONFIG
    return repo_id.split("/")[-1]


def _repo_to_lm_config(repo_id: str) -> str:
    """Map a HuggingFace repo ID to the ACE-Step LM config_path string."""
    if repo_id == MAIN_REPO_ID:
        return _MAIN_REPO_LM_CONFIG
    return repo_id.split("/")[-1]


class TextToMusicNode(SuccessFailureNode):
    """Generate music from a text caption and optional lyrics using the ACE-Step DiT model with optional 5Hz Language Model reasoning."""

    # Class-level model cache keyed by (dit_model, device)
    _dit_handler = None
    _dit_handler_key: tuple | None = None
    _llm_handler = None
    _llm_handler_key: tuple | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="caption",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="Short text prompt describing the desired music style, genre, and mood (max 512 characters)",
            )
        )

        self.add_parameter(
            Parameter(
                name="lyrics",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="[Instrumental]",
                tooltip='Song lyrics with structure tags (e.g. [Verse 1], [Chorus]). Use "[Instrumental]" for no vocals',
            )
        )

        self.add_parameter(
            Parameter(
                name="duration",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=-1.0,
                tooltip="Target audio length in seconds (10-600). Use -1 for automatic duration",
            )
        )

        self._hf_dit_param = HuggingFaceRepoParameter(
            node=self,
            repo_ids=DIT_MODEL_REPO_IDS,
            parameter_name="dit_model",
        )
        self._hf_dit_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="inference_steps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=8,
                tooltip="Number of diffusion steps. Use 8 for turbo models, 32-100 for base/sft models",
            )
        )

        self.add_parameter(
            Parameter(
                name="guidance_scale",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=7.0,
                tooltip="Classifier-free guidance strength (only for non-turbo models)",
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=-1,
                tooltip="Random seed for reproducibility. -1 uses a random seed",
            )
        )

        self.add_parameter(
            Parameter(
                name="bpm",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=0,
                tooltip="Target beats per minute (30-300). Leave 0 for automatic detection",
            )
        )

        self.add_parameter(
            Parameter(
                name="vocal_language",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="unknown",
                tooltip='Vocal language code (e.g. "en", "zh", "ja"). Use "unknown" for auto-detection',
            )
        )

        self.add_parameter(
            Parameter(
                name="enable_lm",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=False,
                tooltip="Enable 5Hz Language Model for Chain-of-Thought metadata and audio code generation",
            )
        )

        self._hf_lm_param = HuggingFaceRepoParameter(
            node=self,
            repo_ids=LM_MODEL_REPO_IDS,
            parameter_name="lm_model",
        )
        self._hf_lm_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="device",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="auto",
                tooltip='Compute device: "auto", "cuda", "mps", or "cpu"',
                traits={Options(choices=DEVICE_CHOICES)},
            )
        )

        self.add_parameter(
            Parameter(
                name="audio",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="AudioUrlArtifact",
                default_value=None,
                tooltip="Generated audio as an AudioUrlArtifact (48kHz, stereo, FLAC)",
            )
        )

        self._create_status_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that required inputs are present."""
        errors = []
        caption = self.parameter_values.get("caption", "")
        if not caption:
            errors.append(ValueError("caption is required"))
        dit_errors = self._hf_dit_param.validate_before_node_run()
        if dit_errors:
            errors.extend(dit_errors)
        return errors if errors else None

    def _get_submodule_root(self) -> str:
        """Return the path to the ace-step-1.5 submodule root."""
        return os.path.join(os.path.dirname(__file__), "ace-step-1.5")

    def _load_dit_handler(self, config_path: str, device: str) -> Any:
        """Load and cache the DiT handler."""
        # DEFERRED IMPORT: import model code here, not at module top level.
        # This only runs after the advanced library has initialized the submodule.
        submodule_root = self._get_submodule_root()
        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)
        from acestep.handler import AceStepHandler

        cache_key = (config_path, device)
        if TextToMusicNode._dit_handler is not None and TextToMusicNode._dit_handler_key == cache_key:
            logger.info("Using cached DiT handler for %s on %s", config_path, device)
            return TextToMusicNode._dit_handler

        logger.info("Initializing DiT handler: %s on %s", config_path, device)
        handler = AceStepHandler()
        handler.initialize_service(
            project_root=submodule_root,
            config_path=config_path,
            device=device,
        )
        TextToMusicNode._dit_handler = handler
        TextToMusicNode._dit_handler_key = cache_key
        logger.info("DiT handler initialized")
        return handler

    def _load_llm_handler(self, lm_config: str, device: str) -> Any:
        """Load and cache the LLM handler."""
        # DEFERRED IMPORT: import model code here, not at module top level.
        submodule_root = self._get_submodule_root()
        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)
        from acestep.llm_inference import LLMHandler

        cache_key = (lm_config, device)
        if TextToMusicNode._llm_handler is not None and TextToMusicNode._llm_handler_key == cache_key:
            logger.info("Using cached LLM handler for %s on %s", lm_config, device)
            return TextToMusicNode._llm_handler

        logger.info("Initializing LLM handler: %s on %s", lm_config, device)
        handler = LLMHandler()
        checkpoint_dir = os.path.join(submodule_root, "checkpoints")
        handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_config,
            backend="auto",
            device=device,
        )
        TextToMusicNode._llm_handler = handler
        TextToMusicNode._llm_handler_key = cache_key
        logger.info("LLM handler initialized")
        return handler

    def process(self) -> AsyncResult[None]:
        """Kick off async inference."""
        device = self.parameter_values.get("device", "auto")
        if device == "auto":
            # The engine detects the device without importing torch, which the process that
            # only edits a workflow does not have.
            device = self.execution_device
        yield lambda: self._run_inference(device)

    def _run_inference(self, device: str) -> None:
        """Run inference (called in background thread via AsyncResult)."""
        # DEFERRED IMPORT: import model code here, not at module top level.
        submodule_root = self._get_submodule_root()
        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        caption = self.parameter_values.get("caption", "")
        lyrics = self.parameter_values.get("lyrics", "[Instrumental]")
        duration = self.parameter_values.get("duration", -1.0)
        dit_repo_id, _ = self._hf_dit_param.get_repo_revision()
        dit_config = _repo_to_dit_config(dit_repo_id)
        inference_steps = self.parameter_values.get("inference_steps", 8)
        guidance_scale = self.parameter_values.get("guidance_scale", 7.0)
        seed = self.parameter_values.get("seed", -1)
        bpm = self.parameter_values.get("bpm", 0)
        vocal_language = self.parameter_values.get("vocal_language", "unknown")
        enable_lm = self.parameter_values.get("enable_lm", False)

        dit_handler = self._load_dit_handler(dit_config, device)

        llm_handler = None
        if enable_lm:
            lm_repo_id, _ = self._hf_lm_param.get_repo_revision()
            lm_config = _repo_to_lm_config(lm_repo_id)
            llm_handler = self._load_llm_handler(lm_config, device)

        params = GenerationParams(
            caption=caption,
            lyrics=lyrics,
            duration=duration,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            bpm=bpm if bpm > 0 else None,
            vocal_language=vocal_language,
            thinking=enable_lm,
            task_type="text2music",
        )

        config = GenerationConfig(
            batch_size=1,
            audio_format="flac",
            use_random_seed=(seed == -1),
        )

        save_dir = tempfile.mkdtemp()
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=save_dir)

        if not result.success:
            raise RuntimeError(f"Music generation failed: {result.error}")

        audio_path = result.audios[0]["path"]
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        filename = f"generated_music_{uuid.uuid4().hex[:8]}.flac"
        url = GriptapeNodes.StaticFilesManager().save_static_file(audio_bytes, filename)
        self.parameter_output_values["audio"] = AudioUrlArtifact(url)

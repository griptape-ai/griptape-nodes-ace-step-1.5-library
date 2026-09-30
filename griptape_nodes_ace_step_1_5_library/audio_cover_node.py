import logging
import os
import sys
import tempfile
import uuid
from typing import Any

from griptape.artifacts import AudioArtifact, AudioUrlArtifact  # AudioArtifact used in _audio_artifact_to_bytes
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

# Maps the main repo ID to the DiT config_path bundled inside it.
# All other repos: config_path = repo_id.split("/")[-1].
_MAIN_REPO_DIT_CONFIG = "acestep-v15-turbo"

MAIN_REPO_ID = "ACE-Step/Ace-Step1.5"

DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]


def _repo_to_dit_config(repo_id: str) -> str:
    """Map a HuggingFace repo ID to the ACE-Step DiT config_path string."""
    if repo_id == MAIN_REPO_ID:
        return _MAIN_REPO_DIT_CONFIG
    return repo_id.split("/")[-1]


class AudioCoverNode(SuccessFailureNode):
    """Generate a cover or style-transfer version of a reference audio file, optionally guided by a new caption and lyrics."""

    # Class-level model cache keyed by (dit_model, device)
    _dit_handler = None
    _dit_handler_key: tuple | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="reference_audio",
                allowed_modes={ParameterMode.INPUT},
                type="AudioArtifact",
                input_types=["AudioArtifact", "AudioUrlArtifact"],
                default_value=None,
                tooltip="Source audio to create a cover from",
            )
        )

        self.add_parameter(
            Parameter(
                name="caption",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="Text description for the new style (leave empty to preserve original style)",
            )
        )

        self.add_parameter(
            Parameter(
                name="lyrics",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="New lyrics for the cover. Leave empty to preserve original lyrics",
            )
        )

        self.add_parameter(
            Parameter(
                name="audio_cover_strength",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=1.0,
                tooltip="How closely to follow the reference audio (0.0=creative freedom, 1.0=close to original)",
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
                tooltip="Number of diffusion steps",
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
                tooltip="Generated cover audio as an AudioUrlArtifact (48kHz, stereo, FLAC)",
            )
        )

        self._create_status_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that required inputs are present."""
        errors = []
        reference_audio = self.parameter_values.get("reference_audio")
        if reference_audio is None:
            errors.append(ValueError("reference_audio is required"))
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
        if AudioCoverNode._dit_handler is not None and AudioCoverNode._dit_handler_key == cache_key:
            logger.info("Using cached DiT handler for %s on %s", config_path, device)
            return AudioCoverNode._dit_handler

        logger.info("Initializing DiT handler: %s on %s", config_path, device)
        handler = AceStepHandler()
        handler.initialize_service(
            project_root=submodule_root,
            config_path=config_path,
            device=device,
        )
        AudioCoverNode._dit_handler = handler
        AudioCoverNode._dit_handler_key = cache_key
        logger.info("DiT handler initialized")
        return handler

    def _audio_artifact_to_bytes(self, artifact: AudioArtifact | AudioUrlArtifact) -> bytes:
        """Extract raw bytes from an AudioArtifact or AudioUrlArtifact."""
        if isinstance(artifact, AudioUrlArtifact):
            return artifact.to_bytes()
        return artifact.value

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

        reference_audio = self.parameter_values.get("reference_audio")
        caption = self.parameter_values.get("caption", "")
        lyrics = self.parameter_values.get("lyrics", "")
        audio_cover_strength = self.parameter_values.get("audio_cover_strength", 1.0)
        dit_repo_id, _ = self._hf_dit_param.get_repo_revision()
        dit_config = _repo_to_dit_config(dit_repo_id)
        inference_steps = self.parameter_values.get("inference_steps", 8)
        seed = self.parameter_values.get("seed", -1)

        if reference_audio is None:
            raise ValueError("reference_audio is required")

        # Write reference audio to a temp file so the model can read it
        audio_bytes = self._audio_artifact_to_bytes(reference_audio)
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp.write(audio_bytes)
            src_audio_path = tmp.name

        try:
            dit_handler = self._load_dit_handler(dit_config, device)

            params = GenerationParams(
                task_type="cover",
                src_audio=src_audio_path,
                caption=caption,
                lyrics=lyrics,
                audio_cover_strength=audio_cover_strength,
                inference_steps=inference_steps,
                seed=seed,
                thinking=False,
            )

            config = GenerationConfig(
                batch_size=1,
                audio_format="flac",
                use_random_seed=(seed == -1),
            )

            save_dir = tempfile.mkdtemp()
            result = generate_music(dit_handler, None, params, config, save_dir=save_dir)

            if not result.success:
                raise RuntimeError(f"Cover generation failed: {result.error}")

            audio_path = result.audios[0]["path"]
            with open(audio_path, "rb") as f:
                output_bytes = f.read()

            filename = f"cover_{uuid.uuid4().hex[:8]}.flac"
            url = GriptapeNodes.StaticFilesManager().save_static_file(output_bytes, filename)
            self.parameter_output_values["audio"] = AudioUrlArtifact(url)

        finally:
            # Clean up temp reference audio file
            if os.path.exists(src_audio_path):
                os.unlink(src_audio_path)

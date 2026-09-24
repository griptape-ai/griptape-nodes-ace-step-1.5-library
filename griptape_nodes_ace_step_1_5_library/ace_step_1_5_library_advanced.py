import logging
import subprocess
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("ace_step_1_5_library")


class AceStep15LibraryAdvanced(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Loading '{library_data.name}' library...")
        if not GriptapeNodes.LibraryManager().is_worker:
            # The submodule populates the execution environment (the `acestep` package),
            # which only the worker imports, so only the worker needs it checked out.
            return
        self._init_submodule()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Finished loading '{library_data.name}' library")

    def _get_library_root(self) -> Path:
        return Path(__file__).parent

    def _init_submodule(self) -> Path:
        """Check out the ACE-Step submodule, which is where the nodes import `acestep` from.

        The submodule stays the source of the model package because ACE-Step's own project
        metadata requires `nano-vllm` from a uv path source, which no index can resolve, so
        naming `acestep` in the manifest's execution set would fail to install. The engine
        clones a library without recursing submodules, so this directory is empty on a fresh
        install and nothing else fills it.
        """
        library_root = self._get_library_root()
        submodule_dir = library_root / "ace-step-1.5"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir
        # The git CLI rather than pygit2: the engine dropped pygit2 (its bundled TLS trust
        # store breaks on some platforms) and requires git on PATH, so it is the one tool
        # guaranteed to be here.
        subprocess.check_call(["git", "-C", str(library_root.parent), "submodule", "update", "--init", "--recursive"])
        if not submodule_dir.exists() or not any(submodule_dir.iterdir()):
            raise RuntimeError(f"Submodule init failed: {submodule_dir}")
        logger.info("Submodule initialized successfully")
        return submodule_dir

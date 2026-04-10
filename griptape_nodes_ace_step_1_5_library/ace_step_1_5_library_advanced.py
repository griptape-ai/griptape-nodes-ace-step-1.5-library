import logging
import subprocess
import sys
from pathlib import Path

import pygit2
from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logger = logging.getLogger("ace_step_1_5_library")


class AceStep15LibraryAdvanced(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Loading '{library_data.name}' library...")
        submodule_path = self._init_submodule()
        if not self._is_installed():
            self._install_from_requirements(submodule_path)
            self._install_package(submodule_path)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Finished loading '{library_data.name}' library")

    def _get_library_root(self) -> Path:
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        root = self._get_library_root()
        if sys.platform == "win32":
            return root / ".venv" / "Scripts" / "python.exe"
        return root / ".venv" / "bin" / "python"

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)
        for sub in repo.submodules:
            sub_path = repo_path / sub.path
            if sub_path.exists() and (sub_path / ".git").exists():
                self._update_submodules_recursive(sub_path)

    def _init_submodule(self) -> Path:
        library_root = self._get_library_root()
        submodule_dir = library_root / "ace-step-1.5"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir
        self._update_submodules_recursive(library_root.parent)
        if not submodule_dir.exists() or not any(submodule_dir.iterdir()):
            raise RuntimeError(f"Submodule init failed: {submodule_dir}")
        logger.info("Submodule initialized successfully")
        return submodule_dir

    def _ensure_pip(self) -> None:
        venv_python = self._get_venv_python_path()
        result = subprocess.run([str(venv_python), "-m", "pip", "--version"], capture_output=True)
        if result.returncode == 0:
            return
        subprocess.check_call([str(venv_python), "-m", "ensurepip", "--upgrade"])

    def _is_installed(self) -> bool:
        """Check if the submodule package is already installed (used to skip re-installation)."""
        venv_python = self._get_venv_python_path()
        result = subprocess.run(
            [str(venv_python), "-c", "import acestep"],
            capture_output=True,
        )
        return result.returncode == 0

    def _install_from_requirements(self, submodule_path: Path) -> None:
        """Install dependencies from the submodule's requirements.txt.

        This preserves platform markers, version pins, and extra-index-url
        directives exactly as the model author specified.
        """
        requirements_file = submodule_path / "requirements.txt"
        if not requirements_file.exists():
            logger.info("No requirements.txt found in submodule, skipping")
            return
        venv_python = self._get_venv_python_path()
        self._ensure_pip()
        logger.info(f"Installing requirements from {requirements_file}...")
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)])
        logger.info("Requirements installed successfully")

    def _install_package(self, submodule_path: Path) -> None:
        """Install the submodule as a Python package (--no-deps since requirements.txt handled deps)."""
        venv_python = self._get_venv_python_path()
        logger.info(f"Installing package from {submodule_path}...")
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--no-deps", str(submodule_path)])
        logger.info("Package installed successfully")

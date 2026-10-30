"""Include shared Python modules and web assets in standalone distributions."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        # A standalone source archive already contains shared/.
        shared = root / "shared"
        if (shared / "server.py").is_file():
            return
        shared = root.parent / "shared"
        if not (shared / "server.py").is_file():
            raise FileNotFoundError(f"Shared modules not found: {shared}")
        build_data.setdefault("force_include", {})[str(shared)] = "shared"

"""Include the shared frontend in both wheels and standalone source archives."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        # Source archives already include web/ through only-include.
        web = root / "web"
        if (web / "index.html").is_file():
            return
        web = root.parent / "web"
        if not (web / "index.html").is_file():
            raise FileNotFoundError(f"Shared frontend not found: {web}")
        build_data.setdefault("force_include", {})[str(web)] = "web"

"""Include the shared frontend in both wheels and standalone source archives."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        # Source archives contain their own copy; checkouts share ../web.
        web = root / "web"
        if not web.is_dir():
            web = root.parent / "web"
        if not (web / "index.html").is_file():
            raise FileNotFoundError(f"Shared frontend not found: {web}")
        build_data.setdefault("force_include", {})[str(web)] = "web"

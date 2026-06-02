from pathlib import Path


class CachePathsConfig:
    """Configure the paths in the app cache"""

    def __init__(self, root: str | Path):

        # Directories
        self.root = Path(root)
        self.img_dataset = self.root / "images"
        self.labelme = self.root / "labelme"

        # Files
        self.labelme_logs = self.root / "labelme.log"

    def mkdir(self):
        for path in self.__dict__.values():
            if isinstance(path, Path) and path.suffix == "":
                path.mkdir(parents=True, exist_ok=True)

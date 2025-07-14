from pathlib import Path
from os import getcwd


class PathVariables:
    __slots__ = (
        "_base_path",
        "model_path",
        "report_path",
        "logs_path",
        "study_path",
        "data_path",
        "h5_file",
    )
    _instance = None

    def __new__(cls, base_path: str | Path = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(base_path)
        return cls._instance

    def _init(self, base_path: str | Path):
        bp = Path(base_path) if base_path else Path(getcwd())
        if not (bp.parent / "data").exists():
            raise FileNotFoundError(
                f"Base path {bp} does not contain the required data directory."
            )

        self._base_path = bp

        out = bp / "outputs"
        out.mkdir(parents=True, exist_ok=True)

        self.model_path = out / "checkpoints"
        self.model_path.mkdir(exist_ok=True)

        self.report_path = out / "reports"
        self.report_path.mkdir(exist_ok=True)

        self.logs_path = out / "logs"
        self.logs_path.mkdir(exist_ok=True)

        self.study_path = out / "studies"
        self.study_path.mkdir(exist_ok=True)

        # Datos
        dp = bp.parent / "data"
        self.data_path = dp
        self.h5_file = dp / "dataset.hdf5"
        #self.h5_file = dp / "dataset_clean_clean.hdf5"
        self.h5_file = dp / "dataset_small_clean25.hdf5"
        #self.h5_file = dp / "dataset_small5_clean.hdf5"

path_vars = PathVariables()

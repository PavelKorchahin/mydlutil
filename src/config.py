# config project
from pathlib import Path
import platform
SYSTEM = platform.system()  # The system type, such as Windows, Linux, or Darwin
ROOT = Path('D:/DefaultProject')  # The root path of the project
DATASET_DIR = Path('D:/Dataset')  # The directory for datasets
PRETRAINED_MODEL_DIR = Path('D:/Pretrained_model')  # The directory for pretrained models
RESULT_MODEL_DIR = Path('D:/Result_model')  # The directory for result models, i.e .pth files
NEED_ACCELERATE_SYSTEM = ('linux', 'unix')  # The system that training will be accelerated, defaultly, training will be accelerated in Linux and Unix system
PATH = [ROOT, DATASET_DIR, PRETRAINED_MODEL_DIR, RESULT_MODEL_DIR] # All the path we need
for path in PATH:
    path.mkdir(parents=True, exist_ok=True)


def get_local_path(path: str | Path) -> Path:
    """
    Get the absolute path from a relative path.
    Args:
        path(str | pathlib.Path): A str or a ``pathlib.Path`` object representing a path.
            If it's an absolute path, returns itself.
            Otherwise, the returned path is the concatenation of the root path of project specified by ``config.ROOT``
            and the given path.
    Returns:
        An absolute path which are a ``pathlib.Path`` object.
    """
    path = Path(path)
    abspath = ROOT / path if not path.is_absolute() else path
    return abspath.resolve()


def is_accelerate() -> bool:
    """
    Check whether to be accelerated when trainning models.
    Defaultly, training will be accelerated in Linux and Unix system.
    If accelerated, number of workers in dataloader will be added and ``torch.cuda.amp`` will be used when training.
    If one wants to train in other system, please add the name of system in the ``NEED_ACCELERATE_SYSTEM`` .
    Returns:
        ``True`` if need to be accelerated when trainning models, otherwise ``False``
    """
    return SYSTEM.lower() in [sys.lower() for sys in NEED_ACCELERATE_SYSTEM]

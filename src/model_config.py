# model configuration
from pathlib import Path
from . import _function as fn
import transformers
from torch import save, load, nn
from torch.nn import functional as F
from pathlib import Path
from .config import PRETRAINED_MODEL_DIR, RESULT_MODEL_DIR
def get_pretrained_model_path(path: str | Path) -> Path:
    """
    Get the absolute path of pretrained model from a relative path.
    Args:
        path(str | pathlib.Path): A str or a ``pathlib.Path`` object representing a path.
            If it's an absolute path, this function returns itself.
            Otherwise, the returned path is the concatenation of the root path of pretrained model specified by ``config.PRETRAINED_MODEL_DIR``
            and the given path.
    """
    path = Path(path)
    abspath = PRETRAINED_MODEL_DIR / path if not path.is_absolute() else path
    return abspath.resolve()

def save_model(model: nn.Module, path: str | Path = None) -> None:
    """
    Save model to the specified path.
    Args:
        model (nn.Module): the model to be saved
        path (str | pathlib.Path, optional): the path of .pth file to save the model.
            It can be an absolute path or a relative path to the  ``config.RESULT_MODEL_DIR``.
            If not specified, the model will be saved to the default path ``config.RESULT_MODEL_DIR``.

    Returns:

    """
    if path is not None:
        path = Path(path)
        abspath = RESULT_MODEL_DIR / path if not path.is_absolute() else path
        path_dir = abspath.parent
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
    else:
        abspath = RESULT_MODEL_DIR
    save(model, abspath.resolve())

def load_model(path: str | Path) -> nn.Module:
    """
    Load model from the specified path.
    Args:
        path (str | pathlib.Path): the path of a .pth file to load the model.
            It can be an absolute path or a relative path to the  ``config.RESULT_MODEL_DIR``.
    """
    path = Path(path)
    abspath = RESULT_MODEL_DIR / path if not path.is_absolute() else path
    return load(abspath.resolve(), weights_only=False)

# The dict of model configuration whose key is the name of model and value is a dict of model version.
# a dict of model version is a dict whose key is the name of model version and value is a dict containing configuration parameters ruled in
# ``generate_pretrained_model`` , ``download_pretrained_model`` in the module ``load_model``
# The configuration parameters  are as follows:
#     - model_dir(str): the directory of the model.
#     - load_func(Callable): the function used to load the model.
#     - model_cls(type): the class of the model.
#     - out(Callable): the function used to process the output of the model.
#     - handle(str): the handle of the model to download.
#     - download_func(Callable): the function used to download the model.
#     - any other parameter: the parameter used to download the model or generate the model object of ``model_cls``.
MODEL_CONFIG = {
    'segformer':{
        'b5_ade':{
            'model_dir': 'SegFormer/segformer-b5-ade',
            'load_func': fn.load_pretrained_model_func_by_huggingface,
            'model_cls': transformers.SegformerForSemanticSegmentation,
            'out': lambda out: F.interpolate(out.logits, scale_factor=4, mode='bilinear', align_corners=False),
            'ignore_mismatched_sizes': True,
            'handle': 'nvidia/segformer-b5-finetuned-ade-512-512',
            'download_func': fn.download_pretrained_model_func_by_huggingface
        },
        'b1_ade':{
            'model_dir': 'SegFormer/segformer-b1-ade',
            'load_func': fn.load_pretrained_model_func_by_huggingface,
            'model_cls': transformers.SegformerForSemanticSegmentation,
            'out': lambda out: F.interpolate(out.logits, scale_factor=4, mode='bilinear', align_corners=False),
            'ignore_mismatched_sizes': True,
            'handle': 'nvidia/segformer-b1-ade-512-512',
            'download_func': fn.download_pretrained_model_func_by_huggingface
        }
    }
}




if __name__ == '__main__':
    ...
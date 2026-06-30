# load pretrained model
from typing import Callable, Type
from torch import nn
import warnings
from pathlib import Path
from .model_config import get_pretrained_model_path, MODEL_CONFIG
from src.mydlutil import _function as fn

warnings.filterwarnings('ignore')


class FunctionModel(nn.Module):
    """
    A wrapper class for a callable object as a module that is a subclass of ``nn.Module`` .
    Args:
        f (Callable): The function to wrap.
    """

    def __init__(self, f: Callable):
        super(FunctionModel, self).__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def download_pretrained_model(
        handles: str | list, names: list[str] | str,
        download_func: Callable[[str, str], None] = fn.download_pretrained_model_func_by_huggingface
) -> None:
    """
        Download pretrained model by ``download_func`` online.
        Args:
            handles (str, list[str]): The list of handles of pretrained models or one pretrained model handle.
                each handle should be meet the specifications of the download fuction given by param ``download_func``

                - str: the handle of one pretrained model to be downloaded, which is equal to [handles].

                - list[str]: The list of handles of pretrained models to be downloaded.

            names(str, list[str]): The list of names of pretrained models or one pretrained model name.
                The pretrained model will be downloaded to the directory ``PRETRAINED_MODEL_DIR/name`` ,
                where ``PRETRAINED_MODEL_DIR`` is the root directory of the pretrained model, i.e. ``config.PRETRAINED_MODEL_DIR`` ,
                and name is ruled as follows:

                - str: the name of one pretrained model to be downloaded, which is equal to [names].
                  In this case, ``handles`` must be a str.

                - list[str]: The list of names of pretrained models to be downloaded.
                  In this case the length of ``names`` should be the same as the length of ``handles`` , namly the number of pretrained models to be downloaded.
                  each name and each handle in ``names`` and ``handles`` must be corresponding one by one.

            download_func(Callable[[str, str], None], optional): The function used to download pretrained models.
                It should be a download function that takes two parameters named handle and out_dir
                where handle is the pretrained model handle and out_dir is the path str of directory that the model will be downloaded to.
                Same goes for that, the path of diectory can be an absolute path or a path relative to the root directory of the pretrained model, i.e. ``config.PRETRAINED_MODEL_DIR``
                the form of ``handle`` must meet the specifications of the download fuction
                Default is ``download_pretrained_model_func_by_huggingface`` in the module ``_function`` ,
                which downloads pretrained model from huggingface, namely ``https://huggingface.co/`` .
                One can visit ``https://hf-mirror.com/`` that the mirror site of huggingface If he/she is in China.
                that is to say, the form of ``handle`` must meet the specifications of ``huggingface_hub.snapshot_download``
                which is the core function for executing the download in the function ``download_pretrained_model_func_by_huggingface`` .
                One can customize a download function that meets the parameters requirements above
        """
    if isinstance(handles, str):
        handles = [handles]
    if isinstance(names, str):
        names = [names]
    assert len(names) == len(handles), 'the length of name and handle should be same'

    for handle, name in zip(handles, names):
        name = Path(name)
        out_dir = get_pretrained_model_path(name)
        out_dir.mkdir(exist_ok=True, parents=True)
        download_func(handle=handle, out_dir=out_dir)


def generate_pretrained_model(
        model_dir: str | Path,
        model_cls: Type[nn.Module],
        load_func: Callable[[str | Path, Type[nn.Module]], nn.Module] = fn.load_pretrained_model_func_by_huggingface,
        out: Callable = fn.identity,
        **kwargs
) -> nn.Module:
    """
    Generate pretrained model by ``load_func``
    Args:
        model_dir (str | ``pathlib.Path`` ): the directory of pretrained model file, specified by atr or a ``pathlib.Path`` object.
            The directory may be an absolute path or a path relative to the root directory of the pretrained model, i.e. ``config.PRETRAINED_MODEL_DIR``
        model_cls (Type[ ``nn.Module`` ]): The class of the pretrained model.
            It must be a subclass of ``nn.Module`` , and its pretrained parameters will be loaded by ``load_func`` .
        load_func (Callable[[str | ``pathlib.Path`` , Type[ ``nn.Module`` ]], ``nn.Module`` ], optional): The function used to load pretrained parameters of the model.
            It should be a load function that takes two postional parameters named model_dir and model_cls and var-keyword parameter kwargs
            where model_dir is the directory of pretrained model , model_cls is the class of pretrained model
            and var-keyword parameter kwargs is other parameters when pretrained parameters loaded.
            Default is ``load_pretrained_model_func_by_huggingface`` in the module ``_function``
            where  pretrained parameters is loaded by the statament ``model_cls.from_pretrained(model_dir, **kwargs)``
            that is standard way of loading pretrained parameters of a model in the ``transformers`` , a library designed to work with ``huggingface`` .
            The module function also offer load function ``load_local_model_by_state_dict``
            which can be used to load pretrained parameters from a .pth file(saved by ``torch.save`` )
            or a dict in the state dict format of the model ``model_cls`` .
            One can customize a load function that meets the parameters requirements above
        out (Callable, optional): The function used to convert the pretrained model's output into the needed format.
            It should be a function or an object of ``nn.Module`` that takes the pretrained model's output as input and return the format one need.
            Default is ``identity`` function in the module ``_function`` that takes x as its input and return x.

        **kwargs: Other parameters needed by ``load_func`` or ``handle`` and ``download_func`` .
            If the directory of pretrained model does not exist or is empty,
            the function will try to download it from the internet by ``handle`` and ``download_func``  when one type 'y'.
            ``handle`` and ``download_func`` see the function ``download_pretrained_model`` .
            In addition to specifying the above three types of parameters, other parameters are not allowed to be passed to kwargs.

    Returns:
        an object of ``nn.Module`` whose form is nn.Sequential(pretrained model, out)( ``out`` has been converted to a model).
    """
    model = None
    model_dir = get_pretrained_model_path(model_dir)
    handle = kwargs.pop('handle', None)
    download_func = kwargs.pop('download_func', None)
    download_func = download_func if download_func is not None else fn.download_pretrained_model_func_by_huggingface
    if not model_dir.exists() or not any(model_dir.iterdir()):
        is_download = input('file does not exist in the path, please noted the configuration of the model '
                            'espacially about data path maybe wrong, '
                            'Do you need to download it, make sure that the pretraibed is not exisit '
                            'instead of wrong  configuration before you decide to download (y/n) ')

        if is_download.lower() == 'y':
            assert handle is not None, 'no handle provided'
            download_func(handle=handle, out_dir=model_dir)
            print('download finished, please configure it in model_config.py correctly and run it again')
            exit(0)
        else:
            exit(0)
    try:
        model = load_func(model_dir=model_dir, model_cls=model_cls, **kwargs)
    except AttributeError:
        raise AttributeError('illegal parameters are passed to kwargs!The parameters passed to kwargs may not be what the load function needs')
    if not isinstance(out, nn.Module):
        out = FunctionModel(out)
    model = nn.Sequential(model, out)
    return model


def load_pretrained_model(
        name: str,
        version:str ='base',
        **kwargs
) -> nn.Module:
    """
    Load pretrained model by ``model_cinfig.MODEL_CONFIG``
    Args:
        name (str): the name of pretrained model that is the key of ``model_cinfig.MODEL_CONFIG`` .
        version (str): the version of pretrained model that is the key in the dict which is the value items in the dict ``model_cinfig.MODEL_CONFIG`` .
        **kwargs: Other parameters needed by load function
            It is not recommended to pass other parameters to kwargs
            All the parameters needed should be listed in the dict ``model_cinfig.MODEL_CONFIG`` completely.

    Returns:
        An object of ``nn.Module``  loaded.

    """
    name = name.lower()
    versiondic = MODEL_CONFIG.get(name, None)
    if versiondic is None:
        raise ValueError(f'the pretrained model {name} does not exist')
    config = versiondic.get(version, None)
    if config is None:
        raise ValueError(f'the {version} of the pretrained model {name} does not exist')
    config.update(**kwargs)
    model = generate_pretrained_model(**config)

    return model


if __name__ == '__main__':
    download_pretrained_model('nvidia/segformer-b1-finetuned-ade-512-512', 'SegFormer/segformer-b1-ade')
    # print(model)

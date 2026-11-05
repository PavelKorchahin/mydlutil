# image processing function
import math
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from PIL.Image import Image
from PIL.Image import open as imgopen
import cv2
import matplotlib.pyplot as plt
from numpy import ndarray,array
import tifffile
from torch import Tensor
import torch.nn.functional as F
from matplotlib import rcParams
from torchvision import transforms
from typing_extensions import Sequence


warnings.filterwarnings('ignore', )
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def img_show(
        imgs: Sequence[Tensor | ndarray | Image] | Tensor | Image | ndarray,
        *titles,
        save: str | Path = None,
        show: bool = True,
        row: int = None,
        **kwargs
) -> plt.Figure:
    """
    Display a list of images, with each row containing as many images as possible.
    Args:
         imgs (Sequence[torch.Tensor | numpy.ndarray | PIL.Image.Image] | torch.Tensor | PIL.Image.Image | numpy.ndarray): A image or a list of images to be displayed.
            Each can be instance(s) of ``torch.Tensor`` , ``numpy.ndarray`` or ``PIL.Image.Image`` .
            If a image given by a instance of ``torch.Tensor`` , the shape of the tensor should be ``[channels, height, width]``.
            If a image given by a instance of ``numpy.ndarray`` , the shape of the array should be ``[height, width, channels]``.
         *titles: titles for each image.
            It can be a list or a tuple of strings or dicts, or strs/dict in the form of postional parameters, namely ``title1,title2,title3...``.
            where ``title_i`` is a str or a dict with key ``title``.
            If a title is given by a dict, the dict should contain the key ``title`` and other keyword arguments
            which can be passed to function that set text properties of the title.
            For example, a dict of a title should be like ``{'title': 'title_name', 'fontweight': 'bold', 'fontfamily': 'SimHei', 'color': 'blue', 'fontsize': 18}`` .
            If a title is given by a str, the properties of the title will be the default.
            Whatever the format of titles, the number of titles should be equal to the number of ``imgs`` and the titles should be matched to ``imgs`` in order.
         save (str | pathlib.Path, optional): The path where the figure should be saved.
            If not specified, the figure will not be saved.
         show (bool, optional): Whether to display the figure when calling the fuction.
            Default is ``True`` .
        row (int, optional): The number of rows of images to be displayed.
            The number of columns will be automatically calculated if specified.
            If not specified, it will be calculated automatically based on the number of images.
        kwargs: Additional keyword arguments which can be passed to function ``matplotlib.pyplot.imshow``.
    Returns:
        The figure which the images are displayed on.
    """
    if titles and isinstance(titles[0], (list, tuple)) and not isinstance(titles[0], str):
        titles = titles[0]
    titles = list(titles)
    for ind in range(len(titles)):
        if isinstance(titles[ind], str):
            titles[ind] = {'title': titles[ind]}
    assert not titles or len(imgs) == len(titles), f'The number of titles ({len(titles)}) does not match the number of images ({len(imgs)})'
    if isinstance(imgs, Tensor) or isinstance(imgs, ndarray) or isinstance(imgs, Image):
        imgs = [imgs]
    num = len(imgs)
    if row is None:
        row = int(math.sqrt(num))
    col = math.ceil(num / row)
    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(10, 10))

    i = 0
    if row == 1:
        axes = array([axes])
        if col == 1:
            axes = array([axes])
    elif col == 1:
        axes = axes.reshape((row, 1))


    for k in range(0, row):
        for j in range(0, col):
            if i >= len(imgs):
                axes[k, j].remove()
                continue
            img = imgs[i]
            if isinstance(img, Tensor) and img.dim() == 3:
                img = img.permute(1, 2, 0).detach().cpu().numpy()

            axes[k, j].imshow(img, **kwargs)
            axes[k, j].axis('off')
            if titles:
                title = titles[i]['title']
                titles[i].pop('title')
                axes[k, j].set_title(title, **titles[i])
            i = i + 1
    fig.tight_layout()
    fig.subplots_adjust()
    if show:
        plt.show()
    if save:
        fig.savefig(save)
    return fig


def open_img(
        filepath: str | Path,
        resize: tuple[int] | int = None,
        to_tensor: bool = True,
        is_tiff: bool = False
) -> Tensor | np.ndarray:
    """
    Open an image file and convert it into a tensor or a numpy array.
    Args:
        filepath (str | pathlib.Path): The absolute path of the image file.
        resize (tuple[int] | int, optional): The target size of the image after resizing if needed.
            If not specified, the image will not be resized.

            - tuple ``(h, w)`` : the target size will be (h, w).

            - int ``n`` : the target size will be (n, n).

        to_tensor (bool, optional): Whether to convert the image to a tensor.

            - ``True`` (default) : the image will be converted to a tensor with shape ``[channels, height, width]`` .

            - ``False`` : the image will be converted to a numpy array with shape ``[height, widyh, channels]`` .

        is_tiff (bool, optional): Whether the image is a tiff file.

    Returns:
        The converted image: a tensor if ``to_tensor`` is ``True`` or a numpy array if ``to_tensor`` is ``False``

    """
    if not is_tiff:
        img = imgopen(filepath)
    else:
        img = tifffile.imread(filepath)
    translst = []
    if resize:
        translst.append(transforms.Resize(resize))
    if to_tensor:
        translst.append(transforms.ToTensor())
    if len(translst) == 0:
        return img
    else:
        trans = transforms.Compose(translst)
        return trans(img)


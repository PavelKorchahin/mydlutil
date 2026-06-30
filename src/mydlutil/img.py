# image processing function
import math
import warnings
from pathlib import Path
from typing import Union
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
        vmin: int = None,
        vmax: int = None
) -> plt.Figure:
    """
    Display a list of images, with each row containing as many images as possible.
    Args:
         imgs (Sequence[torch.Tensor | numpy.ndarray | PIL.Image.Image] | torch.Tensor | PIL.Image.Image | numpy.ndarray): A image or a list of images to be displayed.
            Each can be instance(s) of ``torch.Tensor`` , ``numpy.ndarray`` or ``PIL.Image.Image`` .
            If a image given by a instance of ``torch.Tensor`` , the shape of the tensor should be ``[channels, height, width]``.
            If a image given by a instance of ``numpy.ndarray`` , the shape of the array should be ``[height, width, channels]``.
         *titles: titles for each image.
            It can be a list of strings, or strs in the form of postional parameters, namely ``title1,title2,title3...``.
            Whatever the format of titles, the number of titles should be equal to the number of ``imgs`` and the titles should be matched to ``imgs`` in order.
         save (str | pathlib.Path, optional): The path where the figure should be saved.
            If not specified, the figure will not be saved.
         show (bool, optional): Whether to display the figure when calling the fuction.
            Default is ``True`` .
         vmin (int, optional): The minimum value for the color scale.
            If not specified, it will be shown defaultly.
         vmax (int, optional): The maximum value for the color scale.
            If not specified, it will be shown defaultly.
    Returns:
        The figure which the images are displayed on.
    """
    if titles and isinstance(titles[0], Sequence) and not isinstance(titles[0], str):
        titles = titles[0]
    assert not titles or len(imgs) == len(titles), f'The number of titles ({len(titles)}) does not match the number of images ({len(imgs)})'
    if isinstance(imgs, Tensor) or isinstance(imgs, ndarray) or isinstance(imgs, Image):
        imgs = [imgs]
    num = len(imgs)
    row = int(math.sqrt(num))
    col = math.ceil(num / row)
    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(10, 10))

    i = 0
    if row == 1:
        axes = array([axes])
        if col == 1:
            axes = array([axes])
    for k in range(0, row):
        for j in range(0, col):
            if i >= len(imgs):
                axes[k, j].remove()
                continue
            img = imgs[i]
            if isinstance(img, Tensor) and img.dim() == 3:
                img = img.permute(1, 2, 0).detach().cpu().numpy()
            axes[k, j].imshow(img, vmin=vmin, vmax=vmax)
            if titles:
                axes[k, j].set_title(titles[i], fontsize=8)
            i = i + 1
    fig.tight_layout()
    fig.subplots_adjust()
    if save:
        fig.savefig(save)
    if show:
        plt.show()
    return fig

def open_img(
        filepath: str | Path,
        resize: tuple[int] | int = None,
        to_tensor: bool = True,
        is_tiff: bool = False
) -> Tensor:
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
        The converted image.

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



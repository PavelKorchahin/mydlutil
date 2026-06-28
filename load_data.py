# load data files as a dataset
import pathlib
from pathlib import Path
from typing import Literal, Callable, Sequence, Type
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
from PIL.Image import open as img_open
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from .dataset_config import DATASET_DIR, DATASET_CONFIG
from .img import img_show
from .config import is_accelerate
from . import _function as fn
from .dataset_config import get_dataset_path


class DatasetConfig:
    """
    Define the dataset configuration.
    The dataset can be unlabeled or labeled.
    an instance include all imformation about a specified dataset.

    Notes:
        - all the path mentioned here can be an absolute path or a path relative to the root directory of the dataset, i.e. ``config.DATASET_DIR``
        - 'pixelwised' in the following means the label is a image, mask or something like that
        - If a function need specifying, it had better not be a lambda function or a inner function, that it to say,
          it need defining outer in the file.
          one can define a custom function in the module  ``_function``  where all kinds of functions store specifically
    Args:
        name (str): The name of the dataset.
            Default is 'undefined'.

        get_data (list | str | Path): The source of the features.
            It can be a str, a ``pathlib.Path`` object or a list

            - a str or a ``pathlib.Path`` object to specify the directory of the feature files in the dataset.

            - a list of the paths of feature files in the dataset where each path can be specified by str or ``pathlib.Path`` object.

            - a list of ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` objects representing the each image(feature) in the dataset.
              It's not advised to use this form to specify the feature source which will take up lots of memory.

            - a list of other objects representing the each feature in the dataset.
              In the case, one must specify param ``open_data``  by offering the function that transforms the object in this list to any one of
              ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` objects

        get_label (list | str | Path | Callable | None, optional): The source of the labels.
            It can be None, a str, a list, a ``pathlib.Path`` object or a callable objects

            - None(default):It means the data in dataset is unlabeled.

            - (simple labels) a str or a ``pathlib.Path`` object to specify the '.txt' file that contains the labels
              when the labels are not pixelwised about features(i.e. param ``pixelwise`` is ``False`` ).
              In this case, the labels given by '.txt' file must match up the corresponding features given by param ``get_data`` one by one

            - (simple labels) a list of labels whose type is str when the labels are not pixelwised about features(i.e. param ``pixelwise`` is ``False`` ).

            - (pixelwised labels) a str or a ``pathlib.Path`` object to specify the directory of the label files
              when the labels are pixelwised about features (i.e. param ``pixelwise`` is ``True`` )

            - (pixelwised labels) a list of the paths of labels file where each path can be specified by str or ``pathlib.Path`` object.
              when the labels are pixelwised about features (i.e. param ``pixelwise`` is ``True`` )

            - (pixelwised labels) a list of ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` objects representing the each label in the dataset
              when the labels are pixelwised about features (i.e. param ``pixelwise`` is ``True`` )
              It's not advised to use this form to specify the label source which will take up lots of memory.

            - (pixelwised labels) a list of other objects representing the each label in the dataset.
              In the case, one must specify param ``open_label``  by offering the function that transforms the object in this list to any one of
              ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` objects

            - a callable object that converts each feature path to its corresponding label (if it's simple lable) or label file path (if it's pixelwised lable),
              or converts each feature itself to its corresponding label.
              It's can be useful when the paths is similar between features and their correspongding labels,
              or when converting feature to their correspongding labels just needs a simple function.

            .. note::
                one need ensure the given labels match up the corresponding features given by param ``get_data`` one by one correctly.
                If param ``get_data`` is given by paths ( either specify them directly by a list or specify the directory where they're), they will be sorted by alphabetically.
                If param ``get_data`` is given by the list of any objects representing the feature, they will keep their order.
                So are the param ``get_label`` , and if param ``get_label`` is a callable object , they will not be sorted by alphabetically,
                i.e. they will keep their order with features or paths of feature files given by param ``get_data`` .

        data_pattern (Callable[[str], bool] | str | Path, optional): The form of the paths of feature files.
            It can be a str, a ``pathlib.Path`` object or any callable object mapping str to bool
            Default is a function named ``true`` that return ``True`` whatever the input is, see ``_function.true`` for more details.

            - a str or a ``pathlib.Path`` object to specify the '.txt' file that contains the name of feature files.
              In the case, param ``get_data`` must be specified by paths (either specify them directly by a list or specify the directory where they're)

            - a callable object mapping str to bool to pick out the paths of feature files that makes the callable object return True.
              It only works if param ``get_data`` must be specified by paths (either specify them directly by a list or specify the directory where they're)

        label_pattern (Callable[[str], bool] | str | Path, optional): The form of the paths of label files.
            It can be a str, a ``pathlib.Path`` object or any callable object mapping str to bool as follows:
            Default is a function named ``true`` that return ``True`` whatever the input is, see ``_function.true``  for more details.

            - a str or a ``pathlib.Path`` object to specify the '.txt' file that contains the name of label files.
              In the case, param ``get_label`` must be specified by paths (either specify them directly by a list or specify the directory where they're)

            - a callable object mapping str to bool to pick out the paths of feature files that makes the callable object return True.
              It only works if param ``get_lable`` must be specified by paths (either specify them directly by a list or specify the directory where they're)

        target_suffixes (str | Sequence[str], optional): The suffixe(s) of feature files or label files.
            It can be a str or a sequence of str.
            Default is ``('.jpg', '.jpeg', '.png')``
            It works only if param ``get_data`` or param ``get_lable`` are specified by the directory,
            in the case, one don't have to just pick out the suffixes he/she wants by specifying param ``data_pattern`` or param ``label_pattern`` by a function
            unless he/she have other requrements for the name of their paths.

        label_to_idx(dict, optional): A dict mapping the names of classes to their corresponding class number.
            It can be any of the following key-value pair formats, the key-value pairs must bijections (i.e. one-to-one) in either case:

            - (str, int) or (int, str): str is the name of class, int is corresponding class number.

            - (str, tuple | list) or (tuple, str): it is used when the label images need to map each pixel to its corresponding class name.
              tuple(list is allowed if it is a value) is the channels of a pixel in the label images, str is the name of coresponding class,
              in this case, the class number is the index of class name given by str in param ``class_label`` .
              It works only for preprocessing, that is to say,
              It works only when param ``preprocess`` is ``True`` ,
              and param ``preprocess_func`` is ``_function.map_index`` which is a function resposible for the preprocessing to map each pixel to its corresponding class number.
              See function ``_function.map_index`` for more details.

            - (int, tuple | list) or (tuple, int): it is used when the label images need to map each pixel to its corresponding class number.
              tuple(list is allowed if it is a value) is the channels of a pixel in the label images, int is its corresponding class number.
              It works only for preprocessing, that is to say,
              It works only when param ``preprocess`` is ``True`` ,
              and param ``preprocess_func`` is ``_function.map_index`` which is a function resposible for the preprocessing to map each pixel to its corresponding class number.
              See function ``_function.map_index`` for more details.


        class_label (Sequence[str], optional): The names of classes.
            It can be a sequence of str.
            Default is ``None`` .

            .. note::
                class number must start from 0 and be contigeous.

        num_class (int, optional): The number of data classes.
            If not specifiee, it means the dataset has no classes, or it can be obtained by the length of param ``class_label``  or param `` label_to_idx``

        mode ('train' | 'test' | 'img', optional): The mode of the dataset
            It's allowed to specify the following three values:

            - 'train' (default): it's used for train data. The data under train mode will be augmented by param ``transform`` and normalized by param ``normalize`` ``mean`` and ``std`` .

            - 'test': it's used for test data. The data under test mode won't be augmented and normalized.

            - 'img': it's used for test data which needs to be augmented and normalized. it's source is still teat data specified by function ``generate_dataset``

        pixelwise (bool, optional): whether the labels are pixelwised about features.
            Default is ``False`` .

        target_size (tuple[int, int] | int, optional): The size of the features and labels(if label is pixelwised) by resizing.
            It can be a binary tuple of int or a int
            If not specfied, it means the size of the features and labels(if label is pixelwised) will not resize and keep their original size.
            In the default case,  if each orginal feature (label) have different sizes, it might go wrong when loading data into batches by function ``load_data``
             (i.e. wrap them into a ``torch.utils.data.DataLoader`` object).

            - tuple: specify the height and width of the features and labels(if label is pixelwised) by (height, width).

            - int: it is equivalent to specifying the height and width by (int, int).

            .. note::
                The ways of resizing see the param ``resize_mode`` for more details.

        dset_size (int, optional): The number of items fetched in the dataset.
            If not specfied or large than the number of all items, it will fetch all items specfied by param ``get_data`` and ``get_label`` in the dataset.

        open_data (Callable, optional): The ways to open feature files or transform features specified by param ``get_data`` .
            It can be a callable object that transforms the files or objects specified by param ``get_data``
            to any one of ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` object.

            - When param ``get_data``  is specified by paths(either specify them directly by a list or specify the directory where they're),
              default is ``PIL.Image.open`` if not specified. If feature files in these paths is not allowed to open by ``PIL.Image.open`` ,
              one must specify the attribute by oneself to transform files to ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` object.
              (e.g. if features source is '.tif' files, one can set open_data=tifffile.imread)

            - When param ``get_data``  is specified by the list of any objects other than ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` ,
              one must specify this parameter to transform these objects in the list to any one of ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` objects.

        open_label (Callable, optional): The ways to open label files or transform labels object specified by param ``get_label`` .
            It can be a callable object that transforms the files or objects specified by param ``get_label``
            to any one of ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` object.

            - When param ``get_label``  is specified by paths(either specify them directly by a list or specify the directory where they're),
              default is ``PIL.Image.open`` if not specified. If label files in these paths is not allowed to open by ``PIL.Image.open`` ,
              one must specify this parameter to transform these paths to ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` object.
              (e.g. if label files source is '.tif' files, one can set open_label=tifffile.imread)

            - When param ``get_label``  is specified by the list of any objects other than ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` ,
              one must specify this parameter to transform these objects in the list to any one of ``PIL.Image.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` objects.

            - When the label is not pixelwised, that is to say, each label is a str, one should not specify open_label, even if attr ``get_label`` is a path of '.txt' file.

        device (str): The device to uss when loading the data.
            Default is ``'cpu'`` .
            If you need accelerate training process, one has better not set device= ``'cuda'`` here when generating or loading dataset,
            he/she should set device= ``'cuda'`` that a parameter in ``train.train`` function to accelerate training process.

        resize_mode ('pad' | 'nopad', optional) : The way of resizing the features and labels(if label is pixelwised) when param ``target_size`` is specified.
            It can be ``'pad'`` or ``'nopad'`` .
            Default is ``'pad'`` .

            - ``'pad'`` : the features and labels(if label is pixelwised) will be scaled proportionally
              so the longest side equals the shorter side specified by param ``target_size`` ,
              then paded by zero to the size specified by param ``target_size``

            - ``'nopad'`` : the features and labels(if label is pixelwised) will be directly resized by ``torchvision.transforms.Resize`` object
              to the size specified by param ``target_size`` ,
              where the features will be resized by the mode ``torchvision.transforms.InterpolationMode.BILINEAR``  and
              the label will be resized by the mode ``torchvision.transforms.InterpolationMode.NEAREST`` .

        normalize (bool, optional): Whether to normalize the feature whose param ``mode``  is ``'train'`` or ``'img'`` .
            Default is False.
            Notes that the data whose param ``mode`` is ``'test'`` will not be normalized even if this attribute is True.
            It works only if the number of feature channels is 3.

        mean (Sequence[float], optional): Mean values used for performing normalization if param ``normalize`` is True and param ``mode`` is ``'train'`` or ``'img'`` .
            Default is ``(0.485, 0.456, 0.406)`` that is the mean values of ImageNet.

        std (Sequence[float], optional): Standard deviations used for performing normalization if param ``normalize`` is True and param ``mode`` is ``'train'`` or ``'img'`` .
            Default is ``(0.229, 0.224, 0.225)`` that is the standard deviation values of ImageNet.

        transform (Callable, optional): A list of callable objects or tuples composed of that type used for data augmentation.
            It can be a list whose each element can be a callable objects or a binary tuble whose elements are both callable objects.
            If not specified, keep original data without any data augmentation.

            - If an element in the list is a callable object: the callable object will map the original feature into augmented feature
              In this case, label will not be augmented.

            - If an element in the list is a binary tuple containing two callable objects: the first object in the binary tuple will map the original feature into augmented feature,
              while the second will map the original label into augmented label.
              It's often used for pixelwised label.
              One don't worry that The randomness of data augmentation can result in the data and labels not matching up after augmentation
              If two objects in the binary tuple are instansces of the same class in ``torchvision.transforms`` because of their random seed is fixed to their same index

            .. note::
               - If the transform is specified, augmented data will not include the original data any more.
                 If one wants to keep the original data include in augmented data, he/she should add an element ``torch.nn.Identity()`` or
                 ( ``torch.nn.Identity()`` , ``torch.nn.Identity()`` ) in the list.
               - The number of items fetched in the dataset after augmention will be ``N`` * ``dset_size`` where ``N`` is the length of the transform list.
               - The test dataset whose param ``mode`` is ``'test'`` will not perform any data augmentation even if the transform is specified.

        channels (list | int, optional): The number of channels in each feature.
            It's often used to align channels in each feature when the number of channels in different features varies or not meeting the requirement.
            It can be a list of ints or a int
            If not specified, the number of channels in each feature will not be changed.

            - list of ints: specify the indexs of channels in each feature , where the indexs start from 0.
              If a int denoted as ``n`` in the list is large than the number of channels denoted as ``ch`` in a  feature ,
              it will be viewed as ``n % ch`` .

            - int: it is equivalent to list ``[0, ..., n]`` where ``n`` is the specified int value.

        class_label (Sequence[str], optional): The names of the classes.
            A sequence of strs specifying each class name where the index of name in this sequence is viewed as its corresponding class number.
            If not specified:

            - If param ``get_label`` is None, it means that the data is unlabeled. this parameter won't work(even if it'sspecified).

            - If data is labeled and the param ``label_to_idx`` is specified, the class_label will be set according to mapping relathionship specified by param ``label_to_idx`` .
              In this case class_label can be a list of strs or ints.
              It is usually a list of strs other than the format of items in the ``label_to_idx`` dict is ``(tuble, int)`` or ``(int, tuble)`` , in which case it is a list of int

            - If data is labeled and only the param ``num_class`` is specified, the class_label will be the list ``[0, ..., num_class - 1]``

        label_dims (int, optional): The dimension of the simple label.
            If param ``pixelwise`` is True, default is ``3`` ; else default is ``0`` .
            If the dimension of the label converted by param ``open_label`` is less than tjis value,
            the label will keep inserting dimensions at the position 0 until its dimension is equal to this value.
            note the dimension of the label is not including the batch size demension

        get_index (Callable, optional): A callable object responsible for final processing of features and labels before they're returned in iterations
            It should be a callable object that takes two parameters named res and config as input and returns a data format that one desired to iterate through
            where res is a tuble with which is (feature, label) or (feature, ) depending on whether the data is labeled or not and
            config is the DatasetConfig instance of dataset being loaded.
            It's very useful and flexibe for one to do some other processing on the data before they are iterated through.
            If not specified, the default is ``default_get_index`` ,
            which will map the pixel value above param ``num_class`` in a pixelwised label to ``255`` if param ``pixelwise`` is True and data is labeled.
            The definiton of `` default_get_index`` see the module ``_function`` for more details.

        handle (str, optional): the dataset handle
            It is used to download dataset from this handle by param ``download_func``
            When the data source given by param ``get_data" and ``get_label`` has no data
            In this case, one need to type ``y`` to download it
            It should be meet the specifications of the download fuction given by param ``download_func``
            If not specified, ``FileNotFoundError`` will be raised when no data found in data source.
            It's a record about the imformation about dataset source when migrating data


        download_func (Callable[[str, str], None], optional): A function responsible for downloading the dataset from online platforms.
            It should be a download function that takes two parameters named handle and out_dir
            where handle is the dataset handle and out_dir is the path str of directory that dataset will be downloaded to.
            Same goes for that, the path of diectory can be an absolute path or a path relative to the root directory of the dataset, i.e. ``config.DATASET_DIR``
            the form of param ``handle`` must meet the specifications of the download fuction
            Default is function ``download_dataset_func_by_kaggle`` in the module ``_function`` ,
            which will download the dataset from Kaggle, namely ``https://www.kaggle.com`` ,
            that is to say, the form of param ``handle`` must meet the specifications of ``kagglehub.dataset_download``
            which is the core function for executing the download in the function ``download_dataset_func_by_kaggle`` .
            One can customize a download function that meets the parameters requirements above.


        preprocess (bool, optional): Whether to preprocess the data by I/O operations.
            Default is False.
            If one need the data file itself to be preprocessed and persisted through I/O operations,
            he/she should specify the preprocessing function by param ``preprocess_func`` and set this parameter to ``True``
            .. warning:
                Since it involves I/O operations, one must be extra careful with preprocessing.
                It's best to back up the raw data before one start preprocessing to prevent data loss.
                Preprocessing must only be done when one loads data by this library at the first time.
                once preprocessed, one should set this parameter to ``False`` so that the I/O operations won't be repeated again when loading the same data again.

        preprocess_func (Callabel, optional): The function to preprocess the data.
            It is a function to prepocess the data that takes one parameter named config
            where config is the DatasetConfig instance of dataset being loaded.
            Preprocessing refers to the need to directly modify the original data files through I/O operations, so that these raw data files meet the training specifications.
            If there's no such need for persistence, one shouldn't implement other processing steps by preprocessing function,
            and the param ``preprocess`` should be set to ``False`` .
            The function ``map_index`` in the function in the module ``_function`` is an example of preprocessing function offered in this library.
            It's used for mapping each pixel to its corresponding class number with param ``label_to_idx`` which is a dict with tuples.
            See function ``_function.map_index`` for more details.
            .. warning:
                Since it involves I/O operations, one must be extra careful with preprocessing.
                It's best to back up the raw data before one start preprocessing to prevent data loss.
                Preprocessing must only be done when one loads data by this library at the first time.
                once preprocessed, one should set this parameter to ``False`` so that the I/O operations won't be repeated again when loading the same data again.
        **kwargs: Other arguments received, but it is not recommended to use without specifications if you are not familiar with all the code
            See Also the function ``generate_dataset`` in this module for more details.
    """

    def __init__(
            self,
            name: str = 'undefined',
            get_data: list | str | Path | None = None,
            get_label: list | str | Path | Callable | None = None,
            data_pattern: Callable[[str], bool] | str | Path = fn.true,
            label_pattern: Callable[[str], bool] | str | Path = fn.true,
            target_suffixes: Sequence[str] = ('.jpg', '.jpeg', '.png'),
            label_to_idx: dict = None,
            num_class: int = None,
            mode: Literal['train', 'test', 'img'] = 'train',
            pixelwise: bool = False,
            target_size: int | tuple[int, int] = None,
            dset_size: int = None,
            open_data: Callable | None = None,
            open_label: Callable | None = None,
            device: str = 'cpu',
            resize_mode: Literal['pad', 'nopad'] = 'pad',
            normalize: bool = False,
            mean: Sequence[float] = (0.485, 0.456, 0.406),
            std: Sequence[float] = (0.229, 0.224, 0.225),
            transform=None,
            channels: list | int = None,
            class_label: Sequence[str] = None,
            label_dims: int = None,
            get_index: Callable = fn.default_get_index,
            handle: str = None,
            download_func: Callable[[str, str], None] = fn.download_dataset_func_by_kaggle,
            preprocess: bool = False,
            preprocess_func: str = None,
            **kwargs
    ):
        self.name = name
        self.get_data = get_data
        self.get_label = get_label
        self.data_pattern = data_pattern
        self.label_pattern = label_pattern
        self.target_suffixes = target_suffixes
        self.label_to_idx = label_to_idx
        self.num_class = num_class
        self.mode = mode
        self.pixelwise = pixelwise
        self.target_size = target_size
        self.dset_size = dset_size
        self.open_data = open_data
        self.open_label = open_label
        self.device = device
        self.resize_mode = resize_mode
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.transform = transform
        self.channels = channels
        self.class_label = class_label
        self.label_dims = label_dims
        self.get_index = get_index
        self.handle = handle
        self.download_func = download_func
        self.preprocess = preprocess
        self.preprocess_func = preprocess_func
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        """
        Update the attributes of the instance of ``DatasetConfig``
        Args:
            **kwargs: any parameters in the  ``DatasetConfig``

        """
        self.__dict__.update(kwargs)

    def get_info(self):
        """
        Return the info of an instance, namely its __dict__.
        """
        return self.__dict__

    def __str__(self):
        return self.get_info().__str__()


class GenericDataset(Dataset):
    """
    Process the data in the dataset
    and  iterating through to get the final processed data with the form (feature, label) or (feature, ).
    If the data is unlabeled, or the method ``GenericDataset.get_only_feature`` has been called, it will iterate (feature, ),
    otherwise, it will iterate (feature, label).
    It is a subclass of ``torch.utils.data.Dataset`` .

    Args:
        config (DatasetConfig): The configuration of the dataset, namely an instance of ``DatasetConfig`` .
            If is not specified, an instance of ``DatasetConfig`` will be created by other parameters in kwargs.

        **kwargs: It can take any parameters in the class ``DatasetConfig`` , see the class ``DatasetConfig`` for more details.
            Other parameters is allowed to be specified when param ``config`` has been specified.
            In this case, other specified parameters will update the content of ``config`` .
     """
    train_data = None
    test_data = None
    train_label = None
    test_label = None

    def __init__(self, config: DatasetConfig = None, **kwargs):

        if config is None:
            config = DatasetConfig(**kwargs)
        else:
            config.update(**kwargs)
        if config.mode != 'train' and config.mode != 'test' and config.mode != 'img':
            raise AttributeError('mode should be train, test, or img  ')
        self.name = config.name
        self.mode = config.mode

        if config.mode == 'train':
            self.dset = self.train_data
            self.label_lst = self.train_label
        elif config.mode == 'test' or config.mode == 'img':
            self.dset = self.test_data
            self.label_lst = self.test_label
        pixelwise = config.pixelwise
        self._get_only_feature = False if self.label_lst is not None else True
        self.target_size = config.target_size
        self.dset_size = config.dset_size if config.dset_size and config.dset_size < len(self.dset) else len(self.dset)
        config.dset_size = self.dset_size
        self.transform = [(nn.Identity().to(config.device), nn.Identity().to(config.device))] if pixelwise else [nn.Identity().to(config.device)]
        self.trans_len = len(self.transform)
        self.pixelwise = pixelwise
        self.num_class = config.num_class
        if self.num_class is None:
            if config.label_to_idx is not None:
                self.num_class = len(config.label_to_idx)
            if config.class_label is not None:
                self.num_class = len(config.class_label)
        self.open_data = config.open_data
        self.open_label = config.open_label

        if config.transform is None:
            config.transform = self.transform
        elif config.transform is not None and (config.mode == 'train' or config.mode == 'img'):
            self.transform = config.transform
            self.trans_len = len(config.transform)
            self.dset_size *= self.trans_len
            dset = [data for data in self.dset for _ in range(self.trans_len)]
            if self.label_lst is not None:
                labels = [label for label in self.label_lst for _ in range(self.trans_len)]
            else:
                labels = None
            for trans in self.transform:
                if isinstance(trans, tuple):
                    if hasattr(trans[0], 'to'):
                        trans[0].to(config.device)
                        trans[1].to(config.device)
                else:
                    if hasattr(trans, 'to'):
                        trans.to(config.device)
            self.dset = dset
            self.label_lst = labels

        self.device = config.device
        self.open_data = config.open_data
        self.open_label = config.open_label
        self.target_size = config.target_size
        if isinstance(config.target_size, int):
            self.target_size = (config.target_size, config.target_size)
        self.resize_mode = config.resize_mode
        config.target_size = self.target_size
        if config.normalize:
            assert config.mean and config.std, 'mean and std should not be None'
            self.normalize = transforms.Normalize(mean=config.mean, std=config.std).to(config.device)
        else:
            self.normalize = nn.Identity().to(config.device)
        self.channels = config.channels
        if isinstance(config.channels, int):
            self.channels = list(range(config.channels))
        if self.label_lst is not None:
            if config.class_label is not None:
                self.class_label = config.class_label
            elif isinstance(self.label_lst[0], str) and config.label_to_idx is None:
                self.class_label = list(set(self.label_lst))
                self.label_to_idx = {v: k for k, v in enumerate(self.class_label)}
            elif config.label_to_idx is not None:
                if isinstance(next(iter(config.label_to_idx)), int):
                    config.label_to_idx = dict(sorted(config.label_to_idx.items(), key=lambda item: item[0]))
                elif isinstance(next(iter(config.label_to_idx.values())), int):
                    config.label_to_idx = dict(sorted(config.label_to_idx.items(), key=lambda item: item[1]))

                keys = list(config.label_to_idx.keys())
                values = list(config.label_to_idx.values())
                if isinstance(keys[0], str):
                    self.class_label = keys
                elif isinstance(values[0], str):
                    self.class_label = values
                elif isinstance(keys[0], int):
                    self.class_label = sorted(keys)
                    config.label_to_idx = dict(sorted(config.label_to_idx.items(), key=lambda item: item[0]))
                elif isinstance(values[0], int):
                    self.class_label = sorted(values)
                    config.label_to_idx = dict(sorted(config.label_to_idx.items(), key=lambda item: item[1]))
            else:
                self.class_label = config.class_label = [str(i) for i in range(config.num_class)] if config.num_class else None
        if self.label_lst is not None:
            config.class_label = self.class_label
        self.label_to_idx = config.label_to_idx
        self.label_dims = config.label_dims
        if self.label_dims is None:
            self.label_dims = 3 if self.pixelwise else 0
        config.label_dims = self.label_dims
        self.config = config
        self.get_index = config.get_index
        if config.preprocess:
            assert config.preprocess_func is not None, 'preprocess_func must be specified when param preprocess is True'
            self.preprocess()

    def __getitem__(self, index):
        """
        Get processed item from dataset by index.
        """
        if index > self.dset_size:
            raise StopIteration
        # process feature
        feature = self._feature_to_map(self.dset[index])
        # process label
        label = self._label_map(self.label_lst[index]) if not self._get_only_feature else None
        # data augmentation
        feature, label = self.transform_map(index, feature, label)
        # resize
        if self.target_size:
            feature, label = self.resize(feature, label)
        returns = (feature, label.long()) if not self._get_only_feature else (feature, )
        return self.get_index(returns, self.config)

    def transform_map(self, index, feature, label):
        """
        Data augmentation.
        See param ``DatasetConfig.transform`` for details.
        """
        trans = self.transform[index % self.trans_len]
        if isinstance(trans, tuple):
            torch.manual_seed(index)
            trans_feature = trans[0](feature)
            torch.manual_seed(index)
            trans_label = trans[1](label)
        else:
            trans_feature = trans(feature)
            trans_label = label
        feature = trans_feature
        label = trans_label
        return feature, label

    def resize(self, feature, label):
        """
        Resize the feature and label to the target size.
        The ways to resize see ``DatasetConfig.resize_mode`` for details.
        """
        if self.resize_mode == 'pad':
            target_row, target_col = self.target_size
            min_target_size = min(target_row, target_col)
            h, w = feature.shape[-2:]
            scale_factor = min_target_size / max(h, w)
            h, w = int(h * scale_factor), int(w * scale_factor)
            feature = transforms.Resize((h, w))(feature)
            pad_row, pad_col = target_row - h, target_col - w
            pad_down = pad_up = pad_row // 2
            pad_right = pad_left = pad_col // 2
            if pad_row % 2:
                pad_up += 1
            if pad_col % 2:
                pad_left += 1
            pad_tup = (pad_left, pad_right, pad_up, pad_down)
            feature = F.pad(feature, pad=pad_tup, mode='constant', value=0)
            if self.pixelwise and self.label_lst is not None:
                label = transforms.Resize((h, w), interpolation=InterpolationMode.NEAREST)(label)
                label = F.pad(label, pad=pad_tup, mode='constant', value=255)
            return feature, label
        else:
            feature = transforms.Resize(size=self.target_size, interpolation=InterpolationMode.BILINEAR)(feature)
            feature = feature.squeeze(0)
            if self.pixelwise and self.label_lst is not None:
                label = transforms.Resize(size=self.target_size, interpolation=InterpolationMode.NEAREST)(label)
                label = label.squeeze(0).long()

            feature = feature.clamp(min=0, max=1)
            return feature, label

    def _label_map(self, label):
        """
        Convert label to tensor meeting the requirements specified by the instance of ``DatasetConfig`` .
        If the values in a label are all less than 1, multiply them by 255.
        """
        if self.open_label is not None:
            label = self.open_label(label)
        if isinstance(label, str):
            if self.label_to_idx is not None:
                label = self.label_to_idx[label]
            else:
                try:
                    label = self.class_label.index(label)

                except ValueError:
                    raise ValueError(f'label {label} is not in {self.class_label} , maybe you should offer class_label or label_to_idx.  ')

        if not isinstance(label, torch.Tensor):
            if isinstance(label, np.ndarray) or isinstance(label, Image):
                if isinstance(label, Image):
                    label = np.array(label)
                label = transforms.ToTensor()(label)
            else:
                label = torch.tensor(label)
        label = label.to(self.device)

        while label.dim() < self.label_dims:
            label = label.unsqueeze(0)
        if torch.all(label <= 1) and self.pixelwise:
            label *= 255
        return label.long()

    def _feature_to_map(self, feature):
        """
        Convert feature to tensor meeting the requirements specified by the instance of ``DatasetConfig`` .
        """
        if self.open_data is not None:
            feature = self.open_data(feature)
        if isinstance(feature, tuple):
            feature = feature[0]
        to_tensor = transforms.ToTensor()
        if not isinstance(feature, torch.Tensor):
            feature = to_tensor(feature)
        if self.channels is not None:
            fs = feature.size(0)
            channels = [c % fs for c in self.channels]
            feature = feature[channels]
        feature = self.normalize(feature)
        feature = feature.to(self.device)
        return feature

    def preprocess(self):
        """
        Preprocess the dataset by the processing function given by ``DatasetConfig.preprocess_func``  .
        See param ``DatasetConfig.preprocess_func`` for details.
        Warnings:
            It involves I/O operations and will modify the label files.
        """
        if self.config.preprocess_func is None:
            raise ValueError('preprocess_mode must be specified when param preprocess is True')
        prompt = input('Preprocessing involves I/O operations and will modify the label files.'
                       'Please ensure your data is backed up first. \n'
                       'After all preprocessing is complete(Don\'t forget preprocessing in both train and test .\n), '
                       'set \'preprocess\' to False in the dataset configuration.\n'
                       'Are you sure you want to proceed with preprocessing?(y/n) ')
        if prompt.lower() != 'y':
            exit(0)
        if callable(self.config.preprocess_func):
            self.config.preprocess_func(self.config)

    def __len__(self):
        return self.dset_size

    def get_only_feature(self):
        """
        Return the dataset(self) without labels.
        Once called, the dataset will iterate only the features, namely (feature, ) each time it iterates.
        """
        self._get_only_feature = True
        return self


class ImageDataset(GenericDataset):
    """
    Process data files and convert them to the dataset according to the dataset config.
    If no data found, it will prompt to download the dataset if ``DatasetConfig.handle`` is specified validly.
    It is a subclass of ``GenericDataset``
    See param ``get_data`` / ``get_lable`` / ``data_pattern`` / ``lable_pattern`` / ``handle`` / ``download_func`` in the class ``DatasetConfig`` for details.
    """
    train_data = None
    test_data = None
    train_label = None
    test_label = None

    def __init__(self, config: DatasetConfig = None, **kwargs):

        if config is None:
            config = DatasetConfig(**kwargs)

        self.data = config.get_data
        name = config.name
        get_label = config.get_label
        get_data = config.get_data
        handle = config.handle
        download_func = config.download_func if config.download_func is not None else fn.download_dataset_func_by_kaggle
        if isinstance(config.target_suffixes, str):
            config.target_suffixes = (config.target_suffixes, )
        _data = get_data
        if isinstance(get_data, (str, Path)):
            get_data = get_dataset_path(get_data)
            if str(config.data_pattern).endswith('.txt'):
                config.data_pattern = get_dataset_path(config.data_pattern)
                lst_idx = []
                with open(config.data_pattern, 'r') as f:
                    lst_idx = f.read().splitlines()
                get_data = sorted([str(p) for p in Path(get_data).rglob('*')
                                   if p.is_file() and p.suffix.lower() in config.target_suffixes
                                   and p.stem in lst_idx])
            else:
                get_data = sorted([str(p) for p in Path(get_data).rglob('*')
                                   if p.is_file() and p.suffix.lower() in config.target_suffixes
                                   and config.data_pattern(p.stem)])
            if len(get_data) == 0:
                if handle:
                    is_download = input('file does not exist in the path, please noted its configuration espacially about data path maybe wrong, '
                                        'Do you need to download it, make sure that the dataset is not exisit '
                                        'instead of wrong  configuration before you decide to download (y/n) ')
                    if is_download.lower() == 'y':
                        download_dataset(handles=handle, names=name, download_func=download_func)
                        print('download finished, please configure it in dataset_config.py correctly and run it again')
                        exit(0)
                    else:
                        raise ValueError('no handle provided')
                else:
                    raise FileNotFoundError('file dose not exist in the path')

        if isinstance(get_data, list):
            if isinstance(get_data[0], (np.ndarray, Image)):
                pass
            elif isinstance(get_data[0], (Path, str)):
                get_data = sorted(get_data, key=str)
                config.open_data = config.open_data or img_open
            else:
                assert config.open_data is not None, 'open_data should be specified when get_data is the list of objects other than str|pathlib.Path|Image.Image|np.ndarray'
            if str(config.data_pattern).endswith('.txt') and isinstance(_data, list):
                assert isinstance(get_data[0], (str, Path)), 'config.data_pattern should not be specified by .txt file when get_data is the list of paths'
                config.data_pattern = get_dataset_path(config.data_pattern)
                lst_idx = []
                with open(config.data_pattern, 'r') as f:
                    lst_idx = f.read().splitlines()
                get_data = [e for e in get_data if Path(e).stem in lst_idx or e in lst_idx]
                get_data.sort(key=str)
        self.data = get_data

        _label = get_label
        if isinstance(get_label, (str, Path)):
            get_label = get_dataset_path(get_label)
            if str(config.label_pattern).endswith('.txt') or str(get_label).endswith('.txt'):
                config.label_pattern = get_dataset_path(config.label_pattern)
                lst_idx = []
                with open(config.label_pattern, 'r') as f:
                    lst_idx = f.read().splitlines()
                if config.pixelwise:
                    assert not str(get_label).endswith('.txt'), 'get_label should not be .txt file when pixelwise=True'
                    get_label = sorted([p for p in Path(get_label).rglob('*')
                                        if p.is_file() and p.suffix.lower() in config.target_suffixes
                                        and p.stem in lst_idx], key=str)
                else:
                    get_label = lst_idx
            else:
                get_label = sorted([p for p in Path(get_label).rglob('*')
                                    if p.is_file() and p.suffix.lower() in config.target_suffixes
                                    and config.label_pattern(p.stem)], key=str)
            config.open_label = config.open_label or img_open

        elif isinstance(get_label, Callable):
            get_label = list(map(get_label, self.data))

        if isinstance(get_label, list) and not isinstance(_label, Callable):
            if isinstance(get_label[0], (np.ndarray, Image)):
                pass
            elif isinstance(get_label[0], (Path, str)):
                if config.pixelwise:
                    get_label = sorted(get_label, key=str)
                    config.open_label = config.open_label or img_open
            else:
                assert config.open_label is not None, 'open_data should be specified when get_label is the list of objects other than str|pathlib.Path|Image.Image|np.ndarray'
            if str(config.label_pattern).endswith('.txt') and isinstance(_label, list):
                assert isinstance(get_label[0], (str, Path)), 'config.label_pattern should not be specified by .txt file when get_label is the list of paths'
                config.label_pattern = get_dataset_path(config.data_pattern)
                lst_idx = []
                with open(config.data_pattern, 'r') as f:
                    lst_idx = f.read().splitlines()
                get_label = [e for e in get_label if Path(e) in lst_idx or e in lst_idx]
                get_label.sort(key=str)
        self.label = get_label
        if config.mode == 'train':
            if self.train_data is None:
                self.train_data = self.data
            if self.train_label is None:
                assert self.label is None or len(self.label) == len(self.data), \
                    f'the length of train data and train label must be equal, the length of train data is {len(self.data)} but the length of train label is {len(self.label)}'
                self.train_label = self.label
        else:
            if self.test_data is None:
                self.test_data = self.data
            if self.test_label is None:
                assert self.label is None or len(self.label) == len(self.data), \
                    f'the length of test data and test label must be equal, the length of test data is {len(self.data)} but the length of test label is {len(self.label)}'
                self.test_label = self.label

        super().__init__(config)


class Dset(ImageDataset):
    """
    Define an instance of the dataset.
    It is an instance representing a dataset to be initialized finally.
    It's a subclass of ``ImageDataset``
    """

    def __init__(
            self,
            mode: Literal['train', 'test', 'img'] = 'train',
            target_size=256,
            dset_size=None,
            resize_mode: Literal['pad', 'nopad'] = 'pad',
            normalize=False,
            transform=None,
            device='cuda:0',
            **kwargs
    ):
        config = kwargs.pop('config', None)

        if config is None:
            name = kwargs.pop('name')
            get_train_data = kwargs.pop('get_train_data', None)
            get_test_data = kwargs.pop('get_test_data', None)
            get_train_label = kwargs.pop('get_train_label', None)
            get_test_label = kwargs.pop('get_test_label', None)
            get_data = kwargs.pop('get_data', None)
            get_label = kwargs.pop('get_label', None)
            get_train = kwargs.pop('get_train', None)
            get_test = kwargs.pop('get_test', None)

            get_train_data = get_train_data if get_train_data is not None else get_train if get_train is not None else get_data
            get_test_data = get_test_data if get_test_data is not None else get_test if get_test is not None else get_data
            get_train_label = get_train_label if get_train_label is not None else get_train if get_train is not None else get_label
            get_test_label = get_test_label if get_test_label is not None else get_test if get_test is not None else get_label
            if get_train_data is None and get_test_data is None:
                raise ValueError('get_train_data or get_test_data must be specified')
            num_class = kwargs.pop('num_class', None)
            pixelwise = kwargs.pop('pixelwise', False)
            data_pattern = kwargs.pop('data_pattern', fn.true)
            label_pattern = kwargs.pop('label_pattern', fn.true)
            train_pattern = kwargs.pop('train_pattern', None)
            test_pattern = kwargs.pop('test_pattern', None)
            train_data_pattern = kwargs.pop('train_data_pattern', None)
            train_data_pattern = train_data_pattern if train_data_pattern is not None else train_pattern if train_pattern is not None else data_pattern
            train_label_pattern = kwargs.pop('train_label_pattern', None)
            train_label_pattern = train_label_pattern if train_label_pattern is not None else train_pattern if train_pattern is not None else label_pattern
            test_data_pattern = kwargs.pop('test_data_pattern', None)
            test_data_pattern = test_data_pattern if test_data_pattern is not None else test_pattern if test_pattern is not None else data_pattern
            test_label_pattern = kwargs.pop('test_label_pattern', None)
            test_label_pattern = test_label_pattern if test_label_pattern is not None else test_pattern if test_pattern is not None else label_pattern

            config = DatasetConfig(
                name=name,
                get_data=get_train_data if mode == 'train' else get_test_data,
                get_label=get_train_label if mode == 'train' else get_test_label,
                data_pattern=train_data_pattern if mode == 'train' else test_data_pattern,
                label_pattern=train_label_pattern if mode == 'train' else test_label_pattern,
                num_class=num_class,
                pixelwise=pixelwise,
                mode=mode,
                target_size=target_size,
                dset_size=dset_size,
                resize_mode=resize_mode,
                normalize=normalize,
                transform=transform,
                device=device,
                **kwargs
            )
            if get_test_data is None and get_train_data is not None:
                config.mode = 'train'
                config.get_data = get_train_data
                config.get_label = get_train_label
        super().__init__(config)


class ZipDataset(Dataset):
    """
    Zip together the results iterated from several datasets along a certain dimension in the results.
    It is helpful when one wants to zip a dataset of features  and several datasets of labels together to generate a multi-labeled dataset
    Args:
        dsets: A sequence of instances of ``GenericDataset`` to be zipped together.
        zip_index: The index of the dimension to be zipped together.
            Default is 0 , which usually means that what gets zipped together is the feature
        It's a subclass of ``torch.utils.data.Dataset`` , and it also can be viewed as a subclass of ``GenericDataset`` .

    """

    def __init__(self, dsets: Sequence[GenericDataset], zip_index=0):
        super(ZipDataset, self).__init__()
        self.dsets = dsets
        self.zip_index = zip_index

    def __getitem__(self, index):
        if index > len(self):
            raise StopIteration
        res = []
        for dset in self.dsets:
            res.append(dset[index][self.zip_index].to(dset.device))
        return tuple(res)

    @staticmethod
    def _get_dset_size(x):
        return x.dset_size

    def __len__(self):
        return min(self.dsets, key=ZipDataset._get_dset_size).dset_size


def generate_datadset(
        name,
        get_train_data=None,
        get_test_data=None,
        get_train_label=None,
        get_test_label=None,
        original_size=None,
        num_class=None,
        pixelwise=False,
        get_index=fn.default_get_index,
        train_data_pattern=fn.true,
        test_data_pattern=fn.true,
        train_label_pattern=fn.true,
        test_label_pattern=fn.true,
        **others
) -> Type[Dset]:
    """
    Generate a dataset class(class ``Dset`` ).
    Args:
        name:
        get_train_data: The source of the features for training.
            The ways to use this parameter can be seen in the param ``get_data`` in class ``DatasetConfig`` .
        get_test_data: The source of the features for testing.
            The ways to use this parameter can be seen in the param ``get_data`` in class ``DatasetConfig`` .
        get_train_label: The source of the labels for training.
            The ways to use this parameter can be seen in the param ``get_label`` in class ``DatasetConfig`` .
        get_test_label: The source of the labels for testing.
            The ways to use this parameter can be seen in the param ``get_label`` in class ``DatasetConfig`` .
        original_size: The original size of features (and labels if ``pixelwise`` is ``True`` ).
            The ways to use this parameter can be seen in the param ``target_size`` in class ``DatasetConfig`` .
            In fact, The parameters will be initial value of param ``target_size`` in class ``DatasetConfig`` .
        num_class: see param ``num_class`` in class ``DatasetConfig`` .
        pixelwise: see param ``pixelwise`` in class ``DatasetConfig`` .
        get_index: see param ``get_index`` in class ``DatasetConfig`` .
        train_data_pattern: The pattern for training feature files.
            The ways to use this parameter can be seen param ``data_pattern`` in class ``DatasetConfig`` .
        test_data_pattern: The pattern for testing feature files.
            The ways to use this parameter can be seen param ``data_pattern`` in class ``DatasetConfig`` .
        train_label_pattern: The pattern for training label files.
            The ways to use this parameter can be seen param ``label_pattern`` in class ``DatasetConfig`` .
        test_label_pattern: The pattern for testing label files.
            The ways to use this parameter can be seen param ``label_pattern`` in class ``DatasetConfig`` .
        **others: Other parameters which is the same as parameters in class ``DatasetConfig`` .
            It is als0 allowed to take abbreviations in the following case
            - one can use param ``get_data`` to specify param ``get_train_data`` and ``get_test_data`` simultaneously if they are the same

            - one can use param ``get_label`` to specify param ``get_train_label`` and ``get_test_label`` simultaneously if they are the same

            - one can use param ``get_train`` to specify param ``get_train_data`` and ``get_train_label`` simultaneously if they are the same

            - one can use param ``get_test`` to specify param ``get_test_data`` and ``get_test_label`` simultaneously if they are the same

            - one can use param ``data_pattern`` to specify param ``train_data_pattern`` and ``test_data_pattern`` simultaneously if they are the same

            - one can use param ``label_pattern`` to specify param ``train_label_pattern`` and ``test_label_pattern`` simultaneously if they are the same

            - one can use param ``train_pattern`` to specify param ``train_data_pattern`` and ``train_label_pattern`` simultaneously if they are the same

            - one can use param ``test_pattern`` to specify param ``test_data_pattern`` and ``test_label_pattern`` simultaneously if they are the same
            .. note::
                One shouldn't specify parameters that refer to the same meaning at the same time.
                For example, one can't specify param ``get_train_data`` and ``get_data`` at the same time.
    Returns:
        a class of the dataset, in fact, it's class ``Dset`` encapsulated with configuration specified above.
        The instance of Dset can be initialized by all the parameters ruled in the class ``DatasetConfig``  that have not been specified in this function.
        One can specify the attributes of the dataset itself that won't change in this function
        (e.g. ``get_train_data`` , ``get_test_data`` , ``get_train_label`` , ``get_test_label`` , ``num_class`` , ``pixelwise`` , ``get_index`` , etc.),
        while initializing the attributes that need adjusting in the process of training and evaluating when creating an instance of the Dset.
        (e.g. ``mode`` , ``dset_size`` , ``resize_mode`` , ``normalize`` , ``transform`` , etc.)
    """
    def dataset_instance(
            mode: Literal['train', 'test', 'img'] = 'train',
            target_size=original_size,
            dset_size=None,
            resize_mode: Literal['pad', 'nopad'] = 'pad',
            normalize=False,
            transform=None,
            device='cuda:0',
            **kwargs
    ):
        kwargs.update(others)
        kwargs.update(
            name=name,
            get_train_data=get_train_data,
            get_test_data=get_test_data,
            get_train_label=get_train_label,
            get_test_label=get_test_label,
            num_class=num_class,
            pixelwise=pixelwise,
            get_index=get_index,
            mode=mode,
            target_size=target_size,
            dset_size=dset_size,
            resize_mode=resize_mode,
            normalize=normalize,
            transform=transform,
            device=device
        )
        dset = Dset(**kwargs)
        return dset

    return dataset_instance


def data_show(
        data: GenericDataset,
        start_idx: int = 0,
        num: int = 10,
        vmin: int = None,
        vmax: int = None
) -> plt.Figure:
    """
    Show the data in specified range.
    If the data is unlabled or get only features, it will only show features.
    If the data is simple labeled, it will show features titled their corresponding labels.
    If the data is pixelwise labeled, it will show features and their corresponding labels alternately.
    Args:
        data (GenericDataset): The dataset object to be shown.
        start_idx (int, optional): The starting index of samples in the dataset to be shown.
            Default is 0.
        num (int, optional): The number of samples to be shown.
            Default is 10.
        vmin (int, optional): The minimum value of the data to be shown.It is used to adjust the display range of the data, especially when the pixelwised lable shows.
            If not specified, it will be shown defaultly.
        vmax (int, optional): The maximum value of the data to be shown.It is used to adjust the display range of the data, especially when the pixelwised lable shows.
            If not specified, it will be shown defaultly.
    Returns:
        the ``matplotlib.pyplot.Figure`` object.
    """
    class_label = getattr(data, 'class_label', None)
    data = [data[i] for i in range(start_idx, start_idx + num)]
    features = []
    titles = []
    for feature, *label in data:
        if label:
            label = label[0]

        if isinstance(label, torch.Tensor) and label.dim() >= 2:
            features.extend([feature.cpu(), label.cpu()])
        else:
            features.append(feature.cpu())
            if class_label is not None:
                titles.append(class_label[label.item()])
    return img_show(features, titles, vmin=vmin, vmax=vmax)


def download_dataset(
        handles: str | list[str],
        names: str | list[str],
        download_func: Callable[[str, str], None] = fn.download_dataset_func_by_kaggle
) -> None:
    """
    Download datasets by ``download_func`` online.
    Args:
        handles (str, list[str]): The list of handles of datasets or one dataset handle.

            - str: the handle of one dataset to be downloaded, which is equal to [handles].
              the dataset handle see the param ``handle`` in the class ``DatasetConfig`` .

            - list[str]: The list of handles of datasets to be downloaded.

        names(str, list[str]): The list of names of datasets or one dataset name.
            The datasets will be downloaded to the directory ``DATASET_DIR/name`` ,
            where ``DATASET_DIR`` is the root directory of the dataset, i.e. ``config.DATASET_DIR`` ,
            and name is ruled as follows:

            - str: the name of one dataset to be downloaded, which is equal to [names].
              In this case, ``handles`` must be a str.

            - list[str]: The list of names of datasets to be downloaded.
              In this case the length of ``names`` should be the same as the length of ``handles`` , namly the number of datasets to be downloaded.
              each name and each handle in ``names`` and ``handles`` must be corresponding one by one.

        download_func(Callable[[str, str], None], optional): The function used to download datasets.
            Default is ``download_dataset_func_by_kaggle`` in the module ``_function`` , which downloads datasets from kaggle, namely ``https://www.kaggle.com`` .
            See the param ``download_func`` in the class ``DatasetConfig`` and ``kagglehub.dataset_download`` for more details.

    """
    if isinstance(handles, str):
        handles = [handles]
    if isinstance(names, str):
        names = [names]
    assert len(names) == len(handles), 'the length of name and handle should be same'

    for handle, name in zip(handles, names):
        name = Path(name)
        out_dir = DATASET_DIR / name
        out_dir.mkdir(exist_ok=True, parents=True)
        download_func(handle=handle, out_dir=out_dir)


def get_dataset(nameorconfig: str | dict | DatasetConfig, **kwargs) -> Type[GenericDataset]:
    """
    Get a dataset class that is a subclass of ``GenericDatast`` according to the given name or configuration.
    Args:
        nameorconfig (str | dict | DatasetConfig): the configuration about a dataset.
            It can be a str, a dict, or a DatasetConfig object of the configuration about a dataset.

            - str: A key of the dict ``data_config.DATASET_CONFIG`` where the the configuration about a dataset specfied.

            - dict: A dict stored configuration imformation about the dataset just like the value (is a dict) of the dict ``data_config.DATASET_CONFIG``
              The dict is the key-value pair format for parameters in the function ``generate_datadset`` and class ``DatasetConfig``

            - DatasetConfig: a DatasetConfig object of the configuration about a dataset.
              In this case, it can only generate a dataset class in only one mode of ( ``'train'`` , ``'test'`` , ``'img'`` ).

        **kwargs: The other arguments passed to the function ``generate_datadset`` to generate a class of dataset.
    Returns:
        A dataset class.
        See the function ``generate_datadset`` for more details.
    """
    if isinstance(nameorconfig, str):
        nameorconfig = nameorconfig.lower()
        config = DATASET_CONFIG.get(nameorconfig, None)
    elif isinstance(nameorconfig, DatasetConfig):
        config = nameorconfig.__dict__
        if config is None:
            raise ValueError(f'dataset {nameorconfig} not exist, please download the dataset and configure it.or please offer a dict of a dataset configuration')
    else:
        config = nameorconfig
    config.update(**kwargs)
    return generate_datadset(**config)


def load_data(
        dataset: str | dict | DatasetConfig | Type[GenericDataset],
        batch_size: int = 10,
        drop_last: bool = False,
        only_get_dataset: bool = False,
        only_get_dataloader: bool = False,
        train_map: Callable = fn.identity,
        test_map: Callable = fn.identity,
        dset_map: Callable = fn.identity,
        test_mode: Literal['test', 'img'] = 'test',
        **kwargs
)-> tuple[Type[GenericDataset], tuple[DataLoader, DataLoader] | Type[GenericDataset]] | tuple[DataLoader, DataLoader]:
    """
    Load dataset and return a class that is a subclass of ``GenericDataset``
    and a pair of instances of ``torch.utils.data.DataLoader`` for training and testing.
    It is most commonly used in the training process.
    Args:
        dataset (str | dict | DatasetConfig | Type[GenericDataset]): a dataset class or the configuration about a dataset.
            It can be a str, a dict, or a DatasetConfig object of the configuration about a dataset.

            - str: A key of the dict ``data_config.DATASET_CONFIG`` where the the configuration about a dataset specfied.

            - dict: A dict stored configuration imformation about the dataset just like the value (is a dict) of the dict ``data_config.DATASET_CONFIG``
              The dict is the key-value pair format for parameters in the function ``generate_datadset`` and class ``DatasetConfig``

            - DatasetConfig: a DatasetConfig object of the configuration about a dataset.
              In this case, it can only generate a dataset class that is a subclass of ``GenericDatast``  in only one mode of ( ``'train'`` , ``'test'`` , ``'img'`` ).

            - Type(GenericDataset): a dataset class that is a subclass of ``GenericDatast`` .

        batch_size (int) : The batch size.
            Default is 10.
        drop_last (bool, optional) : Whether to drop the last batch for training and testing if the last batch size do not meet the given batch size.
            Default is False.
        only_get_dataset (bool, optional) : Whether to return only the dataset.
            If set to ``True`` , return only a dataset class ``Dset`` .
            In this case, ``only_get_dataloader`` should not be set to ``True`` .
            Default is False.
        only_get_dataloader (bool, optional) : Whether to return only the dataloaders for training and testing.
            If set to ``True`` , return (train_iter, test_iter) which is a pair of instances of ``torch.utils.data.DataLoader`` for training and testing.
            In this case, ``only_get_dataset`` should not be set to ``True`` .
            Default is False.
        train_map (Callable) : a callable object for processing training data, which can let you do the final processing on the training dataset.
            It should be a callable object which takes an instance of ``GenericDataset`` as input
            and returns an instance of ``torch.utils.data.Dataset`` for processed training data.
            Defaulta is a function named ``identity`` that return x if the input is x, see  ``_function.identity`` for more details
        test_map (Callable) : a callable object for processing testing data, which can let you do the final processing on the testing dataset.
            It should be a callable object which takes an instance of ``GenericDataset`` as input
            and returns an instance of ``torch.utils.data.Dataset`` for processed testing data.
            Default is a function named ``identity`` that return x if the input is x, see  ``_function.identity`` for more details
        dset_map (Callable) : A callable object for processing dataset, which can let you do the final processing the dataset, including training data and testing data.
            It should be a callable object which takes an instance of ``GenericDataset`` as input
            and returns an instance of ``GenericDataset`` for processed data.
            Default is a function named ``identity`` that return x if the input is x, see ``_function.identity`` for more details
            It is not advised to be specified when ``train_map`` or ``test_map`` is specified by a function other than identity.
        test_mode ( 'test' | 'img' ) : The mode of testing data.
            It should be one of ``'test'`` and ``'img'`` .
            Default is ``'test'`` .
            the dataset in the ``'test'`` mode does not perform any augmentation and nor normalization
            while the dataset in the ``'img'`` mode does perform augmentation and normalization according to the configuration about the dataset.
            See the class ``DatasetConfig`` for more details.

        **kwargs: Constructor parameters(other than ``mode`` ) for the dataset object whose class is a subclass of ``GenericDataset`` .
            It's not allowed to pass parameters related to generating the dataset class , such as ``get_train_data`` ``label_pattern`` , etc.
            It's not recommended to pass ``mode`` since train and test ('test' or 'img') will be loaded as a dataloader at the same time when ``mode`` is overridden.

    Returns: The loaded dataset.
        - If ``only_get_dataloader`` is ``False`` and ``only_get_dataset`` is ``False`` (default), return ( ``Dset`` , ( ``train_iter`` , ``test_iter`` ))
          where ``Dset`` is a dataset class that is a subclass of ``GenericDatast``
          ``train_iter`` is an instance of ``torch.utils.data.DataLoader`` for training data
          and ``test_iter`` is an instance of ``torch.utils.data.DataLoader`` for testing data.

        - If ``only_get_dataloader`` is ``True`` , return ( ``train_iter`` , ``test_iter`` )
          where ``train_iter`` is an instance of ``torch.utils.data.DataLoader`` for training data
          and ``test_iter`` is an instance of ``torch.utils.data.DataLoader`` for testing data.

        - If ``only_get_dataset`` is ``True`` , return a dataset class that is a subclass of ``GenericDatast`` .

    """
    if isinstance(dataset, (str, dict, DatasetConfig)):
        dset = get_dataset(dataset)
    else:
        dset = dataset
    dset = dset_map(dset)
    assert not (only_get_dataset and only_get_dataloader), 'only_get_dataset and only_get_dataloader should not be True at the same time'
    if only_get_dataset:
        return dset
    train_data = dset(mode='train', **kwargs)
    kwargs.pop('normalize', False)
    kwargs.pop('transform', None)
    test_data = dset(mode=test_mode, transform=None, normalize=False, **kwargs)
    train_data = train_map(train_data)
    test_data = test_map(test_data)
    if is_accelerate():
        dataloader = (DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=4, pin_memory=True, prefetch_factor=4),
                      DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=4, pin_memory=True, prefetch_factor=4))
    else:
        dataloader = (DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=drop_last),
                      DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=drop_last))

    return (dset, dataloader) if not only_get_dataloader else dataloader


if __name__ == '__main__':
    download_dataset('adhisaikiran/tno-dataset', 'tno-dataset')

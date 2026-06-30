# dataset configuration
from pathlib import Path
import kagglehub
import tifffile
from src.mydlutil import _function as fn
from .config import DATASET_DIR
def get_dataset_path(path) -> Path:
    """
    Get the absolute path of dataset files from a relative path.
    Args:
        path(str | pathlib.Path): A str or a ``pathlib.Path`` object representing a path.
            If it's an absolute path, this function returns itself.
            Otherwise, the returned path is the concatenation of the root path of dataset files specified by ``config.DATASET_DIR``
            and the given path.
    Returns:
        An absolute path which are a ``pathlib.Path`` object.
    """
    path = Path(path)
    abspath = DATASET_DIR / path if not path.is_absolute() else path
    return abspath.resolve()

# a dict of dataset configuration whose key is dataset name and value is a dict containing cofiguration parameters ruled in ``load_data.DatasetConfig``
# see ``load_data.DatasetConfig`` for more details
DATASET_CONFIG = {
    'ade20ksegmentation': {
        'name': 'ADE20KSegmentationDataset',
        'get_train_data': 'ADE20k/ADEChallengeData2016/images/training',
        'get_train_label': 'ADE20k/ADEChallengeData2016/annotations/training',
        'get_test_data': 'ADE20k/ADEChallengeData2016/images/validation',
        'get_test_label': 'ADE20k/ADEChallengeData2016/annotations/validation',
        'original_size': 512,
        'num_class': 151,
        'pixelwise': True,
        'class_label': ('wall', 'building', 'sky', 'floo', 'tree', 'ceiling', 'road', 'bed ',
                        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'doo', 'table', 'mountain',
                        'plant', 'curtain', 'chai', 'ca', 'wate', 'painting', 'sofa', 'shelf', 'house', 'sea',
                        'mirro', 'rug', 'field', 'armchai', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                        'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers',
                        'counte', 'sand', 'sink', 'skyscrape', 'fireplace', 'refrigerato', 'grandstand', 'path',
                        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen doo', 'stairway', 'rive',
                        'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flowe', 'book', 'hill', 'bench',
                        'countertop', 'stove', 'palm', 'kitchen island', 'compute', 'swivel chai', 'boat', 'ba',
                        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'towe', 'chandelie', 'awning',
                        'streetlight', 'booth', 'television receive', 'airplane', 'dirt track', 'apparel', 'pole',
                        'land', 'banniste', 'escalato', 'ottoman', 'bottle', 'buffet', 'poste', 'stage', 'van',
                        'ship', 'fountain', 'conveyer belt', 'canopy', 'washe', 'plaything', 'swimming pool', 'stool',
                        'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                        'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwashe',
                        'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan',
                        'fan', 'pie', 'crt screen', 'plate', 'monito', 'bulletin board', 'showe', 'radiato',
                        'glass', 'clock', 'flag'),
        'channels': 3,
        'download': 'shubhjyot/ade20k',
        'download_mode': 'kaggle',
        'get_index': fn.get_ade_index,
    },
    'cityspaces': {
        'name': 'CityspacesDataset',
        'get_train_data': 'Cityspaces/images/train',
        'get_train_label': 'Cityspaces/gtFine/train',
        'get_test_data': 'Cityspaces/images/val',
        'get_test_label': 'Cityspaces/gtFine/val',
        'original_size': 512,
        'num_class': 29,
        'pixelwise': True,
        'label_pattern': fn.cityspaces_label_pattern,
        'label_to_idx': {
            (0, 0, 0, 255): 'void',
            (111, 74, 0, 255): 'void',
            (81, 0, 81, 255): 'void',
            (128, 64, 128, 255): 'road',
            (244, 35, 232, 255): 'sidewalk',
            (250, 170, 160, 255): 'void',
            (230, 150, 140, 255): 'void',
            (70, 70, 70, 255): 'building',
            (102, 102, 156, 255): 'wall',
            (190, 153, 153, 255): 'fence',
            (180, 165, 180, 255): 'void',
            (150, 100, 100, 255): 'void',
            (150, 120, 90, 255): 'void',
            (153, 153, 153, 255): 'pole',
            (250, 170, 30, 255): 'traffic light',
            (220, 220, 0, 255): 'traffic sign',
            (107, 142, 35, 255): 'vegetation',
            (152, 251, 152, 255): 'terrain',
            (70, 130, 180, 255): 'sky',
            (220, 20, 60, 255): 'person',
            (255, 0, 0, 255): 'rider',
            (0, 0, 142, 255): 'car',
            (0, 0, 70, 255): 'truck',
            (0, 60, 100, 255): 'bus',
            (0, 0, 90, 255): 'void',
            (0, 0, 110, 255): 'void',
            (0, 80, 100, 255): 'train',
            (0, 0, 230, 255): 'motorcycle',
            (119, 11, 32, 255): 'bicycle'
        },
        'class_label': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'void'],
        'download': 'xiaose/cityscapes',
        'download_mode': 'kaggle',
        'preprocess': True,
        'preprocess_mode': fn.map_pixel ,
    },
    'div2k': {
        'name': 'DIV2KDataset',
        'get_train_data': 'div2k/DIV2K_train_HR/DIV2K_train_HR',
        'get_test_data': 'div2k/DIV2K_valid_HR/DIV2K_valid_HR',
        'original_size': 256,
        'handle': 'joe1995/div2k-dataset',
        'download_func': kagglehub.datasets.dataset_download
    },
    'vocsegmentation': {
        'name': 'VOCSegmentationDataset',
        'get_data': 'VOCdevkit/VOC2012/JPEGImages',
        'get_label': 'VOCdevkit/VOC2012/SegmentationClass',
        'train_pattern': 'VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
        'test_pattern': 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
        'num_class': 21,
        'original_size': 500,
        'pixelwise': True,
        'class_label': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'ca', 'cat', 'chai',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonito'],
        'handle': 'zhichengwen/voc2012',
        'download_func': kagglehub.datasets.dataset_download
    },
    'gf':{
        'name': 'GFDataset',
        'get_train_data': 'GF/train/images',
        'get_test_data': 'GF/test/images',
        'get_train_label': 'GF/train/labels',
        'get_test_label': 'GF/test/labels',
        'target_suffixe': '.tif',
        'num_classes': 2,
        'class_label': ['land', 'ocean'],
        'pixelwise': True,
        'original_size': 512,
        'open_data': tifffile.imread,
        'open_label': tifffile.imread,
        'get_index': fn.get_gf_index
    }
}

if __name__ == '__main__':
    ...

# one should definite functions in this module if needed

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import kagglehub
import torch
from torchvision import transforms
from tqdm import tqdm
import huggingface_hub


def cityspaces_label_pattern(x):
    return str(x).endswith('_color')
def get_ade_index(res, *args, **kwargs):
    _, label = res
    label = label - 1
    label = label.masked_fill(label == -1, 254)
    label = label.masked_fill(label == 254, 255)
    return _, label
def get_gf_index(res, *args, **kwargs):
    _, label = res
    label = label.masked_fill(label <= 20, 0)
    label = label.masked_fill(label >= 240, 1)
    return _ , label
def true(*args, **kwargs):
    return True
def identity(x):
    return x
def default_get_index(res, config):
    _, *label = res
    num_class = config.num_class
    if len(label) > 0 and config.pixelwise:
        label = label[0]
        valid_mask = (label >= 0) & (label < num_class) | (label == 255)
        label[~valid_mask] = 255
    return _, label
def map_pixel(config):


    _ ={}
    for class_tup, class_id in config.label_to_idx.items():
        assert isinstance(class_tup, tuple) or isinstance(class_id, (tuple, list)), \
            (f'either key in label_to_idx must be a tuple or value in label_to_idx must be a tuple or a list'
             f'but get key type is {type(class_tup)}, and value type is {type(class_id)}')
        if isinstance(class_tup, str):
            try:
                class_tup, class_id = tuple(class_id), config.class_label.index(class_tup)
            except ValueError:
                raise ValueError(
                    f'{class_tup} is not in {config.class_label} , maybe you should offer class_label correctly.')
        if isinstance(class_id, str):
            try:
                class_id = config.class_label.index(class_id)
            except ValueError:
                raise ValueError(
                    f'{class_id} is not in {config.class_label} , maybe you should offer class_label correctly.')
            if config.class_label[class_id] in {'void', 'ignore', 'unlabeld'}:
                class_id = 255
            _.update({class_tup: class_id})
    config.label_to_idx = _
    it = tqdm(config.label_lst, desc='preprocessing...', leave=False)
    for label in it:
        label_dir = label
        if config.open_label is not None:
            label = config.open_label(label)
        if not isinstance(label, torch.Tensor):
            label = transforms.ToTensor()(label)
        label = label.to(config.device)
        if (label <= 1).all():
            label *= 255
        label = label.long()
        if label.size(0) == 1:
            continue
        for class_tup, class_id in config.label_to_idx.items():
            target_color = torch.tensor(class_tup, device=config.device)
            mask = (label.permute(1, 2, 0).eq(target_color)).all(dim=2).to(config.device)
            label.permute(1, 2, 0)[mask] = class_id
        label = label[0].unsqueeze(0).float() / 255
        transforms.ToPILImage()(label).save(label_dir)
    it.close()
def download_pretrained_model_func_by_huggingface(handle, out_dir):
    huggingface_hub.snapshot_download(handle=handle, local_dir=out_dir)
def download_pretrained_model_func_by_kaggle(handle, out_dir):
    kagglehub.model_download(handle=handle, output_dir=out_dir)
def download_dataset_func_by_kaggle(handle, out_dir):
    kagglehub.dataset_download(handle=handle, output_dir=out_dir, force_download=True)
def load_pretrained_model_func_by_huggingface(model_dir, model_cls, **kwargs):
    model = model_cls.from_pretrained(model_dir, **kwargs)
    return model
def load_local_model_by_state_dict(model_dir, model_cls, **kwargs):
    model = model_cls()
    if isinstance(model_dir, dict):
        model.load_state_dict(model_dir)
    else:
        model.load_state_dict(torch.load(model_dir))
    return model



from cgitb import text
from pathlib import Path
from functools import partial

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
import torch.nn.functional as F

from PIL import Image
import numpy as np

import json

from imagen_pytorch.t5 import t5_encode_text


def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class COCODataset(Dataset):
    def __init__(
        self,
        image_folder,
        annotations_folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        convert_image_to_type = None
    ):
        super().__init__()
        self.image_folder = image_folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{image_folder}').glob(f'**/*.{ext}')]
        ids = [int(str(path).split('_')[-1].split('.')[0]) for path in self.paths]
        self.id_to_path = {idn: path for idn, path in zip(ids, self.paths)}
        with open(f"{annotations_folder}/captions_train2014.json", 'r') as f:
            self.annotations = json.load(f)
            
        convert_fn = partial(convert_image_to, convert_image_to_type) if exists(convert_image_to_type) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        annotation = self.annotations['annotations'][index]     
        caption = annotation['caption']
        text_embeds, text_masks = t5_encode_text([caption], return_attn_mask = True)
        text_embeds = F.pad(input = text_embeds, pad=(0, 0, 0, 256-text_embeds.shape[1], 0, 0), mode='constant', value=0)
        text_embeds = text_embeds[0,:,:]

        path = self.id_to_path[annotation['image_id']]
        img = Image.open(path)
        img = np.array(img)
        
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
            img = np.stack((img,)*3, axis=-1)
        if len(img.shape) == 4:
            img = img[:,:,0,:]


        img = Image.fromarray(img)

        

        return self.transform(img), text_embeds

if __name__ == "__main__":
    COCODataset('../train2014', '../annotations_trainval2014/annotations', image_size = 64)
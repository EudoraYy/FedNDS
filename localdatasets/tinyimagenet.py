import os
import re

import datasets
import torch
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
import numpy as np
import random
# import errno
# from utils import check_exist, makedir_exits_ok, save, load

# path = '/home/yyzhao/GradFL-master-new/data/tinyimagenet/tiny-imagenet-200'
# img_path = os.path.join(path, 'train', 'n03763968', 'images', 'n03763968_347.JPEG')
# print(img_path)
# img = Image.open(img_path)
# transform = transforms.ToTensor()
# img_tensor = transform(img)
# print(img_tensor)
# print(img.mode)

# def check_exist(path):
#     return os.path.exists(path)
#
# def makedir_exits_ok(path):
#     try:
#         os.makedirs(path)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             pass
#         else:
#             raise
#     return
#
# def save(input, path, mode='torch'):
#     dirname = os.path.dirname(path)
#     makedir_exits_ok(dirname)
#     if mode == 'torch':
#         torch.save(input, path)
#     elif mode == 'numpy':
#         np.save(path, input, allow_pickle=True)
#     else:
#         raise ValueError('Not valid save mode')
#     return
#
# def load(path, mode='torch'):
#     if mode == 'torch':
#         return torch.load(path, map_location=lambda storage, loc: storage,weights_only=False)
#     elif mode == 'numpy':
#         return np.load(path, allow_pickle=True)
#     else:
#         raise ValueError('Not valid save mode')
#     return

class TinyImagenet(Dataset):
    '''
    def __init__(self, path, split, subset, transform=None):  # split: train or val
        self.path = path
        self.subset = subset
        self.split = split
        self.transform = transform      # add
        self.class_dict = self.get_class_mapping()   # add
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.class_dict.keys())}   # add
        if not check_exist(self.processed_folder):
            self.process()
        data = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.img_paths = data['img_paths']
        self.target = data[subset]
        self.classes_size = len(self.wnid_to_idx)
        self.classes = self.wnid_to_idx
    '''

    def __init__(self, path, split, subset, transform=None):    #train or valid
        self.path = path
        self.split = split
        self.subset = subset
        self.transform = transform
        # print('Beging--------')
        self.dataset = load_dataset(self.path, split=self.split, trust_remote_code=True)
        self.classes = [i for i in range(200)]
        self.classes_size = len(self.classes)
        self.target = self.dataset['label']

    def __getitem__(self, index):
        img, target = self.dataset[index]['image'], self.dataset[index]['label']
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'img': img, self.subset: target}

    def __len__(self):
        return len(self.dataset)

    ''' 
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'img': img, self.subset: self.target[index]}
    
    def __len__(self):
        return len(self.target)

    @property
    def processed_folder(self):
        return os.path.join(self.path, 'processed')

    def process(self):
        train_img_paths, train_labels = self.read_dir('train')
        test_img_paths, test_labels = self.read_dir('val')
        save({'img_paths': train_img_paths, 'label': train_labels}, os.path.join(self.processed_folder, 'train.pt'))
        save({'img_paths': test_img_paths, 'label': test_labels}, os.path.join(self.processed_folder, 'valid.pt'))
        return

    # Define the class-name mapping.
    def get_class_mapping(self):
        data_dir = os.path.join(self.path, "tiny-imagenet-200")
        wnids_path = os.path.join(data_dir, "wnids.txt")
        words_path = os.path.join(data_dir, "words.txt")

        # Check that the metadata files exist.
        if not all(os.path.exists(p) for p in [wnids_path, words_path]):
            raise FileNotFoundError("Missing dataset metadata files")

        # Load the mapping from class IDs to names.
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f]  # .strip() removes leading and trailing characters (whitespace by default).

        class_dict = {}
        with open(words_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0] in wnids:
                    class_dict[parts[0]] = parts[1].split(',')[0]
        return class_dict

    def read_dir(self, filename):
        img_paths, targets = [], []
        path = os.path.join(self.path, "tiny-imagenet-200", filename)
        if filename == 'train':
            for wnid in os.listdir(path):
                clspath = os.path.join(path, str(wnid), "images")
                for img_name in os.listdir(clspath):
                    img_paths.append(os.path.join(clspath, img_name))
                    targets.append(self.wnid_to_idx[wnid])     # Check the data type of wnid.
        elif filename == 'val':
            target_path = os.path.join(path, 'val_annotations.txt')
            with open(target_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        wnid = parts[1]
                        img_paths.append(os.path.join(path, 'images', img_name))
                        targets.append(self.wnid_to_idx[wnid])
        else:
            raise ValueError('not find {}'.format(filename))
        return img_paths, targets
'''
# if __name__ == '__main__':
#     # path = 'D:\\ProgramCode\\vscodeCode\\eccvgrad\\tinyimagenet'
#     # path = '/home/ubuntu/jyb/GradFL/data/tinyimagenet'
#
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # Crop 80% to 100% of the original image area.
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
#     # val_transform = transforms.Compose([
#     #     transforms.Resize(72),
#     #     transforms.CenterCrop(64),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     # ])
#     path = "/home/naiyuechen/GradFL-master-new/data/tinyimagenet"
#     print(path)
#     dataset = TinyImagenet(path, 'train', 'label', transform)
#     print(len(dataset))

# if __name__ == '__main__':
#     transform = transforms.Compose([
#             transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # Crop 80% to 100% of the original image area.
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])
#     import matplotlib.pyplot as plt
#     dataset = TinyImagenet("/home/naiyuechen/GradFL-master-new/data/tinyimagenet", 'train', 'label', transform)

    # class_indices = {label: [] for label in range(10)}
    # for idx in range(len(dataset)):
    #     label = dataset[idx]['label']
    #     if label < 10:  # Process only the first ten classes.
    #         class_indices[label].append(idx)
    #
    # plt.figure(figsize=(20, 20))
    # for label in range(10):
    #     if class_indices[label]:
    #         selected_idx = random.sample(class_indices[label], k=10)
    #         for i, idx in enumerate(selected_idx):
    #             example = dataset[idx]["img"]
    #             example_np = example.numpy().transpose(1, 2, 0)
    #
    #             plt.subplot(10, 10, label*10+i+1)
    #             plt.imshow(example_np)
    #             plt.title(f"Label: {label}")
    #             plt.axis('off')
    # plt.tight_layout()
    # plt.show()
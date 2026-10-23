import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import check_exist, makedir_exits_ok, save, load
# from data import split_dataset, non_iid, make_dataloader


# Build the dataset from files that have already been extracted.
# output size {'img':[32, 32, 3], 'label': 0-9}  classess: dict()
class CIFAR10(Dataset):
    def __init__(self, cifar_path, split, subset, transform=None):  # split: train or test
        self.path = cifar_path
        self.subset = subset
        self.split = split
        self.transform = transform
        if not check_exist(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[subset]
        self.classes = load(os.path.join(self.processed_folder, 'meta.pt'))

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform is not None:
            img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
            img = self.transform(img)
        return {'img': img, self.subset: target}

    def __len__(self):
        return len(self.target)

    @property
    def processed_folder(self):
        return os.path.join(self.path, 'processed')

    def process(self):
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def make_data(self):
        train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_filenames = ['test_batch']
        train_img, train_label = read_pickle_file(self.path, train_filenames)
        test_img, test_label = read_pickle_file(self.path, test_filenames)
        # train_label, test_label list->dict
        train_target, test_target = {'label': train_label}, {'label': test_label}
        with open(os.path.join(self.path, 'batches.meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            # classes = data['label_names']
        classes = {}
        for i, item in enumerate(data['label_names']):
            classes[i] = item
        # (list, dict), (list, dict), (list)
        return (train_img, train_target), (test_img, test_target), classes


class CIFAR100(CIFAR10):

    def make_data(self):
        train_filenames = ['train']
        test_filenames = ['test']
        train_img, train_label = read_pickle_file(self.path, train_filenames)
        test_img, test_label = read_pickle_file(self.path, test_filenames)
        train_target, test_target = {'label': train_label}, {'label': test_label}
        with open(os.path.join(self.path,  'meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        classes = {}
        for i, item in enumerate(data['fine_label_names']):
            classes[i] = item
        return (train_img, train_target), (test_img, test_target), classes


def read_pickle_file(path, filenames):
    img, label = [], []
    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            img.append(entry['data'])
            label.extend(entry['labels']) if 'labels' in entry else label.extend(entry['fine_labels'])
    img = np.vstack(img).reshape(-1, 3, 32, 32)
    # img = img.transpose(0, 2, 3, 1)
    return img, label


# {'num_cases_per_batch': 10000, 'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_vis': 3072}

# test
'''
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# path = 'D:\\PythonProjects\\federateLearn\\data\\cifar10'
path = 'D:\\github\\GradFL-master\\data\\cifar-10-batches-py'
cifar = CIFAR10(path, 'test', 'label', transform=transform)
# print(type(cifar[0]['img']), type(cifar[0]['label']))
print(type(cifar[0]))
print(type(cifar.img))
print(cifar.img.shape, len(cifar.target), cifar.classes)
print(type(cifar.target))
print(cifar[0]['img'].shape)
'''

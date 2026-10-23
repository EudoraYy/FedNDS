import codecs
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# from utils import check_exists, makedir_exist_ok, save, load
from utils.utils import check_exist, makedir_exits_ok, save, load
# from data import split_dataset, non_iid, make_dataloader


class MNIST(Dataset):
    def __init__(self, mnist_path, split, subset, transform=None):
        self.path = mnist_path
        self.split = split
        self.subset = subset
        self.transform = transform
        if not check_exist(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[self.subset]
        self.classes = load(os.path.join(self.processed_folder, 'meta.pt'))

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform is not None:
            Image.fromarray(self.img[index], mode='L')
            img = self.transform(img)
        return {'img': img, self.subset: target}

    def __len__(self):
        return len(self.img)

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
        train_img = read_image_file(os.path.join(self.path, 'train-images-idx3-ubyte'))
        test_img = read_image_file(os.path.join(self.path, 't10k-images-idx3-ubyte'))
        train_label = read_label_file(os.path.join(self.path, 'train-labels-idx1-ubyte'))
        test_label = read_label_file(os.path.join(self.path, 't10k-labels-idx1-ubyte'))
        train_target, test_target = {'label': train_label}, {'label': test_label}
        classes = {}
        for i in range(10):
            classes[str(i)] = i
        # classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_img, train_target), (test_img, test_target), classes


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
        return parsed


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
        return parsed

#test
'''
path = 'D:\\PythonProjects\\federateLearn\\data\\mnist'
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist = MNIST(path, 'test', 'label', transform=transform)
print(type(mnist[0]))
print(type(mnist.target))
print(mnist[0]['img'].shape)
data_split, label_split = non_iid(mnist, 100, 2)
print(len(data_split), label_split[0])
mnist_dataloader = make_dataloader(mnist)
print(type(mnist_dataloader))
print(len(mnist_dataloader))
for batch in mnist_dataloader:
    print(len(batch))
    print(type(batch))
    print(batch['img'].shape, batch['label'])
    break
iterator = iter(mnist_dataloader)
batch = next(iterator)
print(batch['img'].shape, batch['label'])
'''
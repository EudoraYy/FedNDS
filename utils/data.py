import os
import sys
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import localdatasets
from .config import cfg


def get_transform(dataset, model_name):
    transform = None
    if dataset == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset == 'cifar10':
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'cifar100':
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'tinyimagenet':
        # transform = None    # Have finished in dataset
        transform = transforms.Compose([
        transforms.Resize(224) if model_name == 'vgg16' else transforms.RandomHorizontalFlip(p=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'wikitext2':
        transform = None
    else:
        raise ValueError('Can\'t find {}'.format(dataset))
    return transform


def get_dataset(dataset_name, transform):
    path = os.path.join(sys.path[0], 'data', dataset_name)
    processed = os.path.join(path, 'processed')
    force_reprocess = cfg.get('force_reprocess_dataset', False)
    if os.path.exists(processed) and force_reprocess:
        shutil.rmtree(processed)
        print('Deleted last processed dataset cache!')
    elif os.path.exists(processed):
        print('Using existing processed dataset cache.')
    else:
        print('No processed dataset cache found.')
    dataset = {}
    if dataset_name == 'mnist':
        dataset['train'] = localdatasets.MNIST(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.MNIST(path, 'test', 'label', transform=transform)
    elif dataset_name == 'cifar10':
        dataset['train'] = localdatasets.CIFAR10(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.CIFAR10(path, 'test', 'label', transform=transform)
    elif dataset_name == 'cifar100':
        dataset['train'] = localdatasets.CIFAR100(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.CIFAR100(path, 'test', 'label', transform=transform)
    elif dataset_name == 'tinyimagenet':
        dataset['train'] = localdatasets.TinyImagenet(path, 'train', 'label', transform=transform)
        dataset['test'] = localdatasets.TinyImagenet(path, 'valid', 'label', transform=transform)
    elif dataset_name == 'wikitext2':
        dataset['train'] = localdatasets.WikiText2(path, 'train')
        dataset['test'] = localdatasets.WikiText2(path, 'test')
    else:
        raise ValueError('can\'t find {}'.format(dataset_name))
    return dataset


def get_inferen_data(mode, inferen_batch, class_list, inferen_label, client_dataset, global_dataset):
    inferen_data = None
    if cfg['dataset'] == 'wikitext2':
        if mode not in ['aware', 'awareGrad']:
            return inferen_data
        if inferen_batch == -1:
            inferen_batch = len(client_dataset)
        batch_size = min(inferen_batch, len(client_dataset))
        dataloader = make_dataloader(client_dataset, batch_size, shuffle=False)
    elif mode == 'awareGrad':
        '''
        # using local data
        if inferen_batch == -1:
            inferen_batch = len(dataset)
        '''
        inferen_list = []
        for i in inferen_label:
            inferen_list.extend(class_list[i][:inferen_batch // len(inferen_label)])
        inferen_dataset = SplitDataset(global_dataset, inferen_list)
        dataloader = make_dataloader(inferen_dataset, len(inferen_list))
    elif mode == 'aware':
        if inferen_batch == -1:
            inferen_batch = len(client_dataset)
        dataloader = make_dataloader(client_dataset, inferen_batch if inferen_batch <= len(client_dataset) else len(client_dataset))
    else:
        return inferen_data
    iterator = iter(dataloader)
    inferen_data = next(iterator)
    return inferen_data


def split_dataset(dataset, num_users, data_split_mode):
    global label_splid
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], label_splid = iid(dataset['train'], num_users)
        data_split['test'], _ = iid(dataset['test'], num_users)
    elif data_split_mode == 'non-iid':
        if cfg['dataset'] == 'wikitext2':
            data_split['train'], label_splid = wikitext2_contiguous_noniid(dataset['train'], num_users)
            data_split['test'], _ = wikitext2_contiguous_noniid(dataset['test'], num_users)
        else:
            data_split['train'], label_splid = non_iid(dataset['train'], num_users)
            data_split['test'], _ = non_iid(dataset['test'], num_users)
    return data_split, label_splid


def _make_contiguous_text_blocks(dataset, block_size):
    if dataset.token.dim() == 1:
        num_blocks = dataset.token.numel() // block_size
        if num_blocks < 1:
            raise ValueError('WikiText2 token sequence is shorter than one block')
        dataset.token = dataset.token.narrow(0, 0, num_blocks * block_size).reshape(num_blocks, block_size)
    elif dataset.token.dim() == 2:
        num_blocks = dataset.token.size(0)
    else:
        raise ValueError('WikiText2 token tensor must be 1D or 2D')
    return num_blocks


def wikitext2_contiguous_noniid(dataset, num_users, block_size=None):
    """
    Split WikiText2 by contiguous text blocks.

    The raw token stream is first reshaped into consecutive blocks of length
    ``block_size`` (default: cfg['bptt']). Each client receives a consecutive
    range of block indices, so its local data corresponds to a contiguous
    segment of the corpus rather than a random token sample.
    """
    block_size = cfg['bptt'] if block_size is None else block_size
    num_blocks = _make_contiguous_text_blocks(dataset, block_size)
    if num_users > num_blocks:
        raise ValueError(
            'num_users ({}) is larger than WikiText2 text blocks ({}). '
            'Decrease num_users or block_size.'.format(num_users, num_blocks)
        )

    block_splits = np.array_split(np.arange(num_blocks), num_users)
    data_split = {}
    label_split = []
    for user_idx, block_idx in enumerate(block_splits):
        client_blocks = block_idx.tolist()
        data_split[user_idx] = client_blocks
        if client_blocks:
            label_split.append(torch.unique(dataset.token[client_blocks].reshape(-1)).tolist())
        else:
            label_split[user_idx].append([])
    return data_split, label_split


# data_split: dict{0-num_users: [list]}  label_split: [[list]]
def non_iid(dataset, num_users, shard_per_user=2):
    data_split = {i: [] for i in range(num_users)}
    label_split = []
    shard_per_class = shard_per_user * num_users // len(dataset.classes)
    label_idx_split = {}
    label = dataset.target
    label_split_uni = []
    for i in range(len(label)):
        label_i = label[i]
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)
    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        # leftover = label_idx[-num_leftover:] if num_leftover!=0 else []
        new_label_idx = label_idx[:-num_leftover] if num_leftover != 0 else label_idx
        label_idx_split[label_i] = np.array(new_label_idx).reshape(shard_per_class, -1).tolist()
    if not label_split:
        label_split = list(range(len(dataset.classes))) * shard_per_class
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))]
        label_split = label_split.reshape(num_users, -1).tolist()
        for i in range(len(label_split)):
            label_split[i] = torch.tensor(label_split[i]).unique().tolist()
    for i in range(num_users):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))
    return data_split, label_split


def Dirichlet(dataset, num_users, alpha=0.3):
    data_split = {i: [] for i in range(num_users)}
    label = np.array(dataset.target)
    K = len(np.unique(label))
    N = len(label)
    min_size = 0
    while min_size < K:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            # Prevent individual clients from receiving too many samples.
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions)[:-1] * len(idx_k)).astype(int)
            idx_split = np.split(idx_k, split_points)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]
        min_size = min(len(idx_j) for idx_j in idx_batch)
    label_split = []
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        data_split[j] = idx_batch[j]
        client_labels = np.unique(label[data_split[j]]).tolist()
        label_split.append(client_labels)
    return data_split, label_split

def dataset_class_list(dataset):
    if cfg['dataset'] in ['wikitext2']:
        return {}
    class_list = {i: [] for i in range(len(dataset.classes))}   # TODO: revise this implementation.
    label = dataset.target
    for i in range(len(label)):
        label_i = label[i]
        class_list[label_i].append(i)
    return class_list


def iid(dataset, num_users):
    if cfg['dataset'] in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet']:
        label = torch.tensor(dataset.target)
    elif cfg['dataset'] in ['wikitext2']:
        label = dataset.token
    else:
        raise ValueError('Not valid data name')
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    label_split = {}
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        label_split[i] = torch.unique(label[data_split[i]]).tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split, label_split


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]]


# dataloader API
def make_dataloader(dataset, batch_size=16, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

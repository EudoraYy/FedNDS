import errno
import torch
import random
import numpy as np
import os
import torch.optim as optim
import copy
from collections import OrderedDict
import torch.nn.functional as F

from .config import cfg
from .globals import get_base_params
from models.conv import conv
from models.resnet import resnet18
from models.vgg import vgg16
from models.masked_resnet import masked_resnet18
from models.masked_conv import masked_conv
from models.transformer import transformer

# train_classify begin
def set_control(cfg):
    args = get_base_params()
    for k, v in args.items():
        if v is not None:
            cfg[k] = v
    if cfg['mode'] == 'fiarse':
        cfg['fiarse_aggregation_effective'] = cfg.get('fiarse_aggregation', 'original')


def set_seed(seed):
    torch.manual_seed(seed)  # Set the CPU random seed for reproducibility.
    torch.cuda.manual_seed(seed)  # Set the GPU random seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model(model_name, dataset, model_rate):
    if model_name == 'resnet18':
        hidden_size = [64, 128, 256, 512]
        num_blocks = [2, 2, 2, 2]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = resnet18(shape, hidden_size, num_blocks=num_blocks, num_classes=num_class, model_rate=model_rate,
                         model_name=model_name)
    elif model_name == 'resnet34':
        hidden_size = [64, 128, 256, 512]
        num_blocks = [3, 4, 6, 3]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = resnet18(shape, hidden_size, num_blocks=num_blocks, num_classes=num_class, model_rate=model_rate,
                         model_name=model_name)
    elif model_name == 'conv':
        hidden_size = [64, 128, 256, 512]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = conv(shape, hidden_size, num_class, model_rate=model_rate)
    elif model_name == 'vgg16':
        hidden_size = [32, 32, 64, 64, 128, 512]
        num_classes = len(dataset['train'].classes)
        model = vgg16(hidden_size, num_classes, model_rate=model_rate)
    elif model_name == 'masked_conv':
        hidden_size = [64, 128, 256, 512]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = masked_conv(shape, hidden_size, num_class, model_rate=model_rate)
    elif model_name == 'masked_resnet18':
        hidden_size = [64, 128, 256, 512]
        num_blocks = [2, 2, 2, 2]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = masked_resnet18(shape, hidden_size, num_blocks=num_blocks, num_classes=num_class, model_rate=model_rate,
                         model_name=model_name)
    elif model_name == 'masked_resnet34':
        hidden_size = [64, 128, 256, 512]
        num_blocks = [3, 4, 6, 3]
        shape = dataset['train'][0]['img'].shape
        num_class = len(dataset['train'].classes)
        model = masked_resnet18(shape, hidden_size, num_blocks=num_blocks, num_classes=num_class, model_rate=model_rate,
                         model_name=model_name)
    elif model_name == 'transformer':
        model = transformer(model_rate)
    else:
        raise ValueError('Dont have this Model: {}'.format(model_name))
    return model


def make_optimizer(model, lr):
    if cfg['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if cfg['scheduler_name'] == 'None':
        # Multiply the learning rate by gamma (default: 0.1) only at epoch 65535.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['factor'])
    elif cfg['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    elif cfg['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs']['global'],
                                                         eta_min=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['factor'],
                                                         patience=cfg['patience'], verbose=True,
                                                         threshold=cfg['threshold'], threshold_mode='rel',
                                                         min_lr=cfg['min_lr'])
    elif cfg['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr'], max_lr=10 * cfg['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def generate_fix_submodel_rate_list():
    # Sample candidate rates from cfg['submodel_rate'] using cfg['probs'].
    # Draw N=cfg['numbers'] rates to create a fixed-length submodel rate list.
    model_rate_list = cfg['submodel_rate']
    model_rate_list_probs = torch.tensor(cfg['probs'])
    m = torch.distributions.Categorical(probs=model_rate_list_probs)
    samples = m.sample(sample_shape=torch.Size([cfg['numbers']]))
    model_rate_list_fixed = torch.tensor(model_rate_list)[samples]
    model_rate_list_fixed = model_rate_list_fixed.tolist()
    return model_rate_list_fixed


def generate_submodel_rate(mode, sub_model_rate_list, user_idx):
    if mode == 'dynamic':
        m = torch.distributions.Categorical(probs=torch.tensor(cfg['probs']))
        return cfg['submodel_rate'][m.sample()]
    elif mode == 'fix':
        return sub_model_rate_list[user_idx]
    elif mode == 'real':
        m = torch.distributions.Categorical(probs=torch.tensor(cfg['probs']))
        base_rate = cfg['submodel_rate'][m.sample()]
        rate_client_list = [base_rate + cfg['alpha'], base_rate, base_rate - cfg['alpha']]
        if base_rate == 1:
            client_prob = [0, 0.01, 0.99]
            m_client = torch.distributions.Categorical(probs=torch.tensor(client_prob))
        elif base_rate == cfg['alpha']:
            client_prob = [0, 1, 0]
            m_client = torch.distributions.Categorical(probs=torch.tensor(client_prob))
        else:
            client_prob = [0.01, 0.98, 0.01]
            m_client = torch.distributions.Categorical(probs=torch.tensor(client_prob))
        return rate_client_list[m_client.sample()]
    else:
        raise ValueError('Can not match mode {}'.format(mode))


def get_fiarse_model_params(global_model, client_model_idx, client_model):
    client_model_params = OrderedDict()
    client_model_masks = OrderedDict()
    client_modules = dict(client_model.named_modules())
    # Default to use_tcbgd=True when the configuration key is absent.
    use_tcbgd = cfg.get('fiarse_use_tcbgd', True)

    for k, v in global_model.state_dict().items():
        if k in client_model_idx:
            threshold = client_model_idx[k]
            mask = torch.abs(v) >= threshold
            client_model_params[k] = copy.deepcopy(v * mask)
            client_model_masks[k] = mask.detach().clone()

            # Copy the same threshold into the client MaskedConv2d/MaskedLinear modules.
            # Keep using the FIARSE mask and TCB-GD during subsequent local forward and backward passes.
            if k.endswith('.weight') or k.endswith('.bias'):
                module_name, tensor_name = k.rsplit('.', 1)
                module = client_modules.get(module_name)
                if hasattr(module, 'set_threshold'):
                    fixed_mask = None if use_tcbgd else mask
                    module.set_threshold(
                        threshold,
                        bern=use_tcbgd,
                        param_name=tensor_name,
                        fixed_mask=fixed_mask,
                    )
        else:
            client_model_params[k] = copy.deepcopy(v)
            client_model_masks[k] = torch.ones_like(v, dtype=torch.bool)

    return client_model_params, client_model_masks


def get_model_params(global_model, client_model_idx, client_model=None, mode=None):
    if mode == 'fiarse':
        if client_model is None:
            raise ValueError('client_model is required for FIARSE parameter extraction')
        return get_fiarse_model_params(global_model, client_model_idx, client_model)

    client_model_params = OrderedDict()
    for k, v in global_model.state_dict().items():
        if k in client_model_idx:
            if v.dim() > 1:
                client_model_params[k] = copy.deepcopy(v[torch.meshgrid(client_model_idx[k], indexing='ij')])
            else:
                client_model_params[k] = copy.deepcopy(v[client_model_idx[k]])
        else:
            raise NameError('Can\'t match {}'.format(k))
    return client_model_params


def get_fiarse_active_ratio(client_model_masks, threshold_keys):
    active = 0
    total = 0
    for key in threshold_keys:
        mask = client_model_masks[key]
        active += torch.count_nonzero(mask).item()
        total += mask.numel()
    return active / total if total > 0 else 1.0


def load_params_to_client_model(global_model, client_model, inference_data, mode, device, rate, select_mode,
                                label_mask, clients_models_shape, user_idx, current_round, global_gradients,
                                clients_models_masks=None, client_order=0):
    global client_model_params
    if mode == 'aware':
        global_model.to(device)
        with torch.no_grad():
            if cfg['dataset'] == 'wikitext2':
                inference_data = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in inference_data.items()
                }
            else:
                inference_data = inference_data['img'].to(device)
            global_model.get_idx_aware(inference_data, rate, select_mode)
            global_model.cpu()
        client_model_idx = OrderedDict()
        for k, v in global_model.idx.items():
            temp = []
            if isinstance(global_model.idx[k], tuple):
                for index in range(len(global_model.idx[k])):
                    temp.append(global_model.idx[k][index].to('cpu'))
            else:
                temp.append(global_model.idx[k].to('cpu'))
            client_model_idx[k] = (copy.deepcopy(temp[0]), copy.deepcopy(temp[1])) if len(temp) > 1 else temp[0]
    elif mode == 'awareGrad':
        gradient = get_model_gradient(global_model, inference_data, label_mask, device)
        global_model.get_idx_aware_grad(rate, select_mode, gradient)
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'randGrad':
        if current_round == 1:  # Select the submodel randomly in the first round.
            global_model.get_idx_rand(rate)
            client_model_idx = copy.deepcopy(global_model.idx)
        else:  # Use gradients to select the submodel from the second round onward.
            global_model.get_idx_rand_grad(rate, select_mode, global_gradients, current_round)
            client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'fiarse':
            # FIARSE uses rate ** 2 as the parameter retention ratio, approximately matching neuron-based extraction.
            global_model.get_mask_fiarse(rate ** 2, use_tcbgd=cfg.get('fiarse_use_tcbgd', True))
            # client_model_idx is actually a threshold dictionary.
            client_model_idx = {k: v.detach().clone() for k, v in global_model.thresholds.items()}
    elif mode == 'roll':
        global_model.get_idx_roll(rate, roll_offset=global_model.roll + client_order)
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'rand':
        global_model.get_idx_rand(rate)
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'ecover':
        global_model.get_idx_ecover(rate, cfg['model_name'])
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'hetero':
        global_model.get_idx_hetero(rate)
        client_model_idx = copy.deepcopy(global_model.idx)
    elif mode == 'fedavg':
        global_model.get_idx_hetero(1)
        client_model_idx = copy.deepcopy(global_model.idx)
    else:
        raise ValueError('Not valid model name')
    try:
        if mode == 'fiarse':
            # Save each FIARSE client mask each round; reuse it for active-ratio logging and aggregation.
            client_model_params, client_model_masks = get_model_params(global_model, client_model_idx, client_model, mode)
            if clients_models_masks is not None:
                clients_models_masks[int(user_idx)] = copy.deepcopy(client_model_masks)
        else:
            client_model_params = get_model_params(global_model, client_model_idx, client_model, mode)
    except NameError as e:
        print(e)
    global_model.clear_idx()
    client_model.load_state_dict(client_model_params)
    clients_models_shape[int(user_idx)] = copy.deepcopy(client_model_idx)


# train_classify end



# get_gradient from global model using inference data
def get_model_gradient(model, inference_data, label_mask, device):
    model.zero_grad()
    model.to(device)
    input = inference_data['img'].to(device)
    target = inference_data['label'].to(device)

    gradients = OrderedDict()
    handles = []

    def save_gradient(module, grad_in, grad_out, layer_name):
        '''
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            # if self.reshape_transform is not None:
            #     grad = self.reshape_transform(grad)
            gradients[layer_name] = grad.cpu().detach()
        '''
        gradients[layer_name] = torch.abs(grad_out[0].cpu())

    for name, module in model.named_modules():
        if 'conv' in name or 'linear' in name:
            handles.append(module.register_full_backward_hook(
                lambda module, grad_in, grad_out, layer_name=name: save_gradient(module, grad_in, grad_out, layer_name)))

    output = model(input)
    output = output.masked_fill(label_mask == 0, 0)
    loss = F.cross_entropy(output, target)
    loss.backward()
    for handle in handles:
        handle.remove()
    model.cpu()
    return gradients


def get_correct_seq_gradients(gradients):
    temp = list(gradients.items())[::-1]
    temp_seq, shortcut = [], []
    for k, v in temp:
        param_name = k.split('.')[-1]
        if param_name == 'shortcut':
            shortcut.append((k, v))
        elif param_name == 'conv2':
            temp_seq.append((k, v))
            temp_seq.append(shortcut[-1])
        else:
            temp_seq.append((k, v))
    return OrderedDict(temp_seq)


def check_exist(path):
    return os.path.exists(path)


def makedir_exits_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch', protocol=2):
    dirname = os.path.dirname(path)
    makedir_exits_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage,weights_only=False)   # Default weights_only to False to allow loading models and custom classes.
    elif mode == 'numpy':
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return





def compute_fisher_per_neuron(fisher_param):
    # Convert to neuron/channel-level values.
    fisher_neuron = {}

    for name, f in fisher_param.items():

        # Conv: [out_c, in_c, k, k]
        if len(f.shape) == 4:
            fisher_neuron[name] = f.view(f.size(0), -1).sum(dim=1)

        # Linear: [out, in]
        elif len(f.shape) == 2:
            fisher_neuron[name] = f.sum(dim=1)

        # Bias: [out]
        elif len(f.shape) == 1:
            fisher_neuron[name] = f

    return fisher_neuron


def print_all_thresholds(model):
    print("self.conv.threshold:", model.conv.threshold.item())
    for i in range(1,5):
        layer = getattr(model, f'layer{i}')
        for j, block in enumerate(layer):
            print(f"layer{i}.{j}.conv1.threshold:", block.conv1.threshold.item())
            print(f"layer{i}.{j}.conv2.threshold:", block.conv2.threshold.item())
            if hasattr(block.shortcut, 'threshold'):
                print(f"layer{i}.{j}.shortcut.threshold:", block.shortcut.threshold.item())


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def batchify(dataset, batch_size):
    # Reshape a continuous token stream into a 2D batch so each column forms a continuous training sequence.
    num_batch = len(dataset) // batch_size
    dataset.token = dataset.token.narrow(0, 0, num_batch * batch_size)
    dataset.token = dataset.token.reshape(batch_size, -1)
    return dataset


def process_dataset(dataset, split):
    if cfg['data_name'] in ['wikitext2']:
        cfg['vocab'] = dataset['train'].vocab
        cfg['num_tokens'] = len(dataset['train'].vocab)
        dataset[split] = batchify(dataset[split], cfg['batchsize'])
    else:
        raise ValueError('Not valid data name')
    return

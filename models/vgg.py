import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

if __name__ == '__main__':
    from utils import init_param, Scaler
else:
    from .utils import init_param, Scaler


def conv_layer(chann_in, chann_out, k_size, p_size, scaler_rate):
    layer = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size, bias=False)),
        ('sclar', Scaler(scaler_rate)),
        ('bn', nn.BatchNorm2d(chann_out, momentum=None, track_running_stats=False)),
        ('relu', nn.ReLU())
    ]))
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, scaler_rate=1):
    layers = [('name{}'.format(i), conv_layer(in_list[i], out_list[i], k_list[i], p_list[i], scaler_rate)) for i in
              range(len(in_list))]
    # layers += [('maxpool', nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s))]
    return nn.Sequential(OrderedDict(layers))


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(OrderedDict([
        ('linear', nn.Linear(size_in, size_out)),
        ('bn', nn.BatchNorm1d(size_out, momentum=None, track_running_stats=False)),
        ('relu', nn.ReLU())
    ]))
    return layer


class VGG16(nn.Module):
    def __init__(self, hidden_size, n_classes=10, scaler_rate=1):
        super(VGG16, self).__init__()
        l1, l2, l3, l4, l5, fc_hidden = hidden_size
        self.roll = 0
        self.idx = OrderedDict()
        self.scaler = Scaler(scaler_rate)

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, l1], [l1, l1], [3, 3], [1, 1], scaler_rate)
        self.layer2 = vgg_conv_block([l1, l2], [l2, l2], [3, 3], [1, 1], scaler_rate)
        self.layer3 = vgg_conv_block([l2, l3, l3], [l3, l3, l3], [3, 3, 3], [1, 1, 1], scaler_rate)
        self.layer4 = vgg_conv_block([l3, l4, l4], [l4, l4, l4], [3, 3, 3], [1, 1, 1], scaler_rate)
        self.layer5 = vgg_conv_block([l4, l5, l5], [l5, l5, l5], [3, 3, 3], [1, 1, 1], scaler_rate)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * l5, fc_hidden)
        self.layer7 = vgg_fc_layer(fc_hidden, fc_hidden)

        # Final layer
        self.linear = nn.Linear(fc_hidden, n_classes)

    def forward(self, x):
        out = F.max_pool2d(self.layer1(x), kernel_size=2, stride=2)
        out = F.max_pool2d(self.layer2(out), kernel_size=2, stride=2)
        out = F.max_pool2d(self.layer3(out), kernel_size=2, stride=2)
        out = F.max_pool2d(self.layer4(out), kernel_size=2, stride=2)
        vgg16_features = F.max_pool2d(self.layer5(out), kernel_size=2, stride=2)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.linear(out)

        return out

    def get_idx_aware(self, input, rate, topmode):
        out = input
        start_channels, first_channels = torch.arange(3), torch.tensor([])
        layer_name = 'layer'
        layer_index = ['1', '2', '3', '4', '5']
        for i in layer_index:
            cur_layer_name = layer_name + i
            layer = self.get_attar(cur_layer_name)
            sublayer_name = 'name'
            for index, module in enumerate(layer):
                cur_sublayer_name = sublayer_name + str(index)
                cur_name = cur_layer_name + '.' + cur_sublayer_name
                out = module(out)
                first_channels = get_topk_index(out, int(rate * module.conv.weight.shape[0]), topmode)
                self.idx[cur_name + '.conv.weight'] = (first_channels, start_channels)
                self.idx[cur_name + '.bn.weight'], self.idx[cur_name + '.bn.bias'] = first_channels, first_channels
                start_channels = first_channels
            out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        # first_channels = torch.arange(int(out.size(1) * rate))
        temp = [list(range(i, i+49)) for i in first_channels]
        first_channels = []
        for item in temp:
            first_channels.extend(item)
        first_channels = torch.tensor(first_channels)
        out = self.layer6(out)
        second_channels = get_topk_index(out, int(rate * self.layer6.linear.weight.shape[0]), topmode)
        self.idx['layer6.linear.weight'], self.idx['layer6.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer6.bn.weight'], self.idx['layer6.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        out = self.layer7(out)
        second_channels = get_topk_index(out, int(rate * self.layer7.linear.weight.shape[0]), topmode)
        self.idx['layer7.linear.weight'], self.idx['layer7.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer7.bn.weight'], self.idx['layer7.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_aware_grad(self, rate, select_mode, gradient):
        start_channels, first_channels = torch.arange(3), torch.tensor([])
        layer_name = 'layer'
        layer_index = ['1', '2', '3', '4', '5']
        for i in layer_index:
            cur_layer_name = layer_name + i
            layer = self.get_attar(cur_layer_name)
            sublayer_name = 'name'
            for index, module in enumerate(layer):
                cur_sublayer_name = sublayer_name + str(index)
                cur_name = cur_layer_name + '.' + cur_sublayer_name
                first_channels = get_topk_index(gradient[cur_name + '.conv'], int(rate * module.conv.weight.shape[0]),
                                                select_mode)
                self.idx[cur_name + '.conv.weight'] = (first_channels, start_channels)
                self.idx[cur_name + '.bn.weight'], self.idx[cur_name + '.bn.bias'] = first_channels, first_channels
                start_channels = first_channels
        # out = out.view(out.size(0), -1)
        # first_channels = torch.arange(int(out.size(1) * rate))
        temp = [list(range(i, i + 49)) for i in first_channels]
        first_channels = []
        for item in temp:
            first_channels.extend(item)
        first_channels = torch.tensor(first_channels)
        second_channels = get_topk_index(gradient['layer6.linear'], int(rate * self.layer6.linear.weight.shape[0]),
                                         select_mode)
        self.idx['layer6.linear.weight'], self.idx['layer6.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer6.bn.weight'], self.idx['layer6.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        second_channels = get_topk_index(gradient['layer7.linear'], int(rate * self.layer7.linear.weight.shape[0]),
                                         select_mode)
        self.idx['layer7.linear.weight'], self.idx['layer7.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer7.bn.weight'], self.idx['layer7.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_rand_grad(self, rate, select_mode, gradient):
        start_channels, first_channels = torch.arange(3), torch.tensor([])
        layer_name = 'layer'
        layer_index = ['1', '2', '3', '4', '5']
        for i in layer_index:
            cur_layer_name = layer_name + i
            layer = self.get_attar(cur_layer_name)
            sublayer_name = 'name'
            for index, module in enumerate(layer):
                cur_sublayer_name = sublayer_name + str(index)
                cur_name = cur_layer_name + '.' + cur_sublayer_name
                first_channels = get_topk_index(gradient[cur_name + '.conv.weight'], int(rate * module.conv.weight.shape[0]),
                                                select_mode)
                self.idx[cur_name + '.conv.weight'] = (first_channels, start_channels)
                self.idx[cur_name + '.bn.weight'], self.idx[cur_name + '.bn.bias'] = first_channels, first_channels
                start_channels = first_channels
                # out = out.view(out.size(0), -1)
                # first_channels = torch.arange(int(out.size(1) * rate))
        temp = [list(range(i, i + 49)) for i in first_channels]
        first_channels = []
        for item in temp:
            first_channels.extend(item)
        first_channels = torch.tensor(first_channels)
        second_channels = get_topk_index(gradient[param_name + '.conv1.weight'], int(rate * blc.conv1.weight.shape[0]),
                                                 select_mode)
        self.idx['layer6.linear.weight'], self.idx['layer6.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer6.bn.weight'], self.idx['layer6.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        second_channels = torch.roll(torch.arange(self.layer7.linear.weight.shape[0]),
                                     shifts=self.roll % self.layer7.linear.weight.shape[0], dims=-1)[
                          : int(rate * self.layer7.linear.weight.shape[0])]
        self.idx['layer7.linear.weight'], self.idx['layer7.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer7.bn.weight'], self.idx['layer7.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])



        #         blc = self.get_attar(param_name)
        #         second_channels = get_topk_index(gradient[param_name + '.conv1.weight'], int(rate * blc.conv1.weight.shape[0]),
        #                                          select_mode)
        #         self.idx[param_name + '.conv1.weight'] = (second_channels, first_channels)
        #         self.idx[param_name + '.n1.weight'], self.idx[param_name + '.n1.bias'] = second_channels, second_channels
        #
        #         third_channels = get_topk_index(gradient[param_name + '.conv2.weight'], int(rate * blc.conv2.weight.shape[0]),
        #                                         select_mode)
        #         self.idx[param_name + '.conv2.weight'] = (third_channels, second_channels)
        #         self.idx[param_name + '.n2.weight'], self.idx[param_name + '.n2.bias'] = third_channels, third_channels
        #         self.idx[param_name + '.shortcut.weight'] = (third_channels, first_channels)
        #         first_channels = third_channels
        # self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        # self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_roll(self, rate, roll_offset=None):
        roll = self.roll if roll_offset is None else roll_offset
        start_channels, first_channels = torch.arange(3), torch.tensor([])
        layer_name = 'layer'
        layer_index = ['1', '2', '3', '4', '5']
        for i in layer_index:
            cur_layer_name = layer_name + i
            layer = self.get_attar(cur_layer_name)
            sublayer_name = 'name'
            for index, module in enumerate(layer):
                cur_sublayer_name = sublayer_name + str(index)
                cur_name = cur_layer_name + '.' + cur_sublayer_name
                first_channels = torch.roll(torch.arange(module.conv.weight.shape[0]),
                                            shifts=roll % module.conv.weight.shape[0], dims=-1)[
                                 : int(rate * module.conv.weight.shape[0])]
                self.idx[cur_name + '.conv.weight'] = (first_channels, start_channels)
                self.idx[cur_name + '.bn.weight'], self.idx[cur_name + '.bn.bias'] = first_channels, first_channels
                start_channels = first_channels
        # out = out.view(out.size(0), -1)
        # first_channels = torch.arange(int(out.size(1) * rate))
        temp = [list(range(i, i + 49)) for i in first_channels]
        first_channels = []
        for item in temp:
            first_channels.extend(item)
        first_channels = torch.tensor(first_channels)
        second_channels = torch.roll(torch.arange(self.layer6.linear.weight.shape[0]),
                                     shifts=roll % self.layer6.linear.weight.shape[0], dims=-1)[
                          : int(rate * self.layer6.linear.weight.shape[0])]
        self.idx['layer6.linear.weight'], self.idx['layer6.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer6.bn.weight'], self.idx['layer6.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        second_channels = torch.roll(torch.arange(self.layer7.linear.weight.shape[0]),
                                     shifts=roll % self.layer7.linear.weight.shape[0], dims=-1)[
                          : int(rate * self.layer7.linear.weight.shape[0])]
        self.idx['layer7.linear.weight'], self.idx['layer7.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer7.bn.weight'], self.idx['layer7.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_rand(self, rate):
        start_channels, first_channels = torch.arange(3), torch.tensor([])
        layer_name = 'layer'
        layer_index = ['1', '2', '3', '4', '5']
        for i in layer_index:
            cur_layer_name = layer_name + i
            layer = self.get_attar(cur_layer_name)
            sublayer_name = 'name'
            for index, module in enumerate(layer):
                cur_sublayer_name = sublayer_name + str(index)
                cur_name = cur_layer_name + '.' + cur_sublayer_name
                first_channels = torch.randperm(module.conv.weight.shape[0])[: int(rate * module.conv.weight.shape[0])]
                self.idx[cur_name + '.conv.weight'] = (first_channels, start_channels)
                self.idx[cur_name + '.bn.weight'], self.idx[cur_name + '.bn.bias'] = first_channels, first_channels
                start_channels = first_channels
        # out = out.view(out.size(0), -1)
        # first_channels = torch.arange(int(out.size(1) * rate))
        temp = [list(range(i, i + 49)) for i in first_channels]
        first_channels = []
        for item in temp:
            first_channels.extend(item)
        first_channels = torch.tensor(first_channels)
        second_channels = torch.randperm(self.layer6.linear.weight.shape[0])[
                          :int(rate * self.layer6.linear.weight.shape[0])]
        self.idx['layer6.linear.weight'], self.idx['layer6.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer6.bn.weight'], self.idx['layer6.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        second_channels = torch.randperm(self.layer7.linear.weight.shape[0])[
                          :int(rate * self.layer7.linear.weight.shape[0])]
        self.idx['layer7.linear.weight'], self.idx['layer7.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer7.bn.weight'], self.idx['layer7.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_idx_hetero(self, rate):
        start_channels, first_channels = torch.arange(3), torch.tensor([])
        layer_name = 'layer'
        layer_index = ['1', '2', '3', '4', '5']
        for i in layer_index:
            cur_layer_name = layer_name + i
            layer = self.get_attar(cur_layer_name)
            sublayer_name = 'name'
            for index, module in enumerate(layer):
                cur_sublayer_name = sublayer_name + str(index)
                cur_name = cur_layer_name + '.' + cur_sublayer_name
                first_channels = torch.arange(int(rate * module.conv.weight.shape[0]))
                self.idx[cur_name + '.conv.weight'] = (first_channels, start_channels)
                self.idx[cur_name + '.bn.weight'], self.idx[cur_name + '.bn.bias'] = first_channels, first_channels
                start_channels = first_channels
        # out = out.view(out.size(0), -1)
        # first_channels = torch.arange(int(out.size(1) * rate))
        temp = [list(range(i, i + 49)) for i in first_channels]
        first_channels = []
        for item in temp:
            first_channels.extend(item)
        first_channels = torch.tensor(first_channels)
        second_channels = torch.arange(int(rate * self.layer6.linear.weight.shape[0]))
        self.idx['layer6.linear.weight'], self.idx['layer6.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer6.bn.weight'], self.idx['layer6.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        second_channels = torch.arange(int(rate * self.layer7.linear.weight.shape[0]))
        self.idx['layer7.linear.weight'], self.idx['layer7.linear.bias'] = (second_channels,
                                                                            first_channels), second_channels
        self.idx['layer7.bn.weight'], self.idx['layer7.bn.bias'] = second_channels, second_channels
        first_channels = second_channels

        self.idx['linear.weight'] = (torch.arange(self.linear.weight.shape[0]), first_channels)
        self.idx['linear.bias'] = torch.arange(self.linear.weight.shape[0])

    def get_attar(self, attar_name):
        names = attar_name.split('.')
        obj = self
        for x in names:
            obj = getattr(obj, x)
        return obj

    def clear_idx(self):
        self.idx.clear()


def get_topk_index(x, k, topmode):
    if topmode == 'absmax':
        if x.dim() > 2:  # conv
            temp = torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:  # linear
            temp = torch.sum(x, dim=0)
            return torch.topk(temp, k)[1]
    elif topmode == 'l2':
        if x.dim() > 2:  # conv
            temp = torch.norm(x, p=2, dim=(1, 2, 3))
            return torch.topk(temp, k)[1]
        else:  # linear
            return torch.topk(x, k)[1]
    elif topmode == 'probs':
        if x.dim() > 2:
            temp = torch.sum(x, dim=(0, 2, 3))
            probs = torch.abs(temp) / torch.sum(torch.abs(temp))
        else:
            probs = torch.abs(x) / torch.sum(torch.abs(x))
        samples = torch.multinomial(probs, num_samples=k, replacement=False)
        return samples
    elif topmode == 'absmin':
        if x.dim() > 2:
            temp = -torch.sum(x, dim=(0, 2, 3))
            return torch.topk(temp, k)[1]
        else:
            return torch.topk(-x, k)[1]
    else:
        raise ValueError('no method!')


def vgg16(hidden_size, num_classes, model_rate=1):
    # hidden_size = [64, 128, 256, 512, 512, 4096]
    hidden_size = [int(item * model_rate) for item in hidden_size]
    model = VGG16(hidden_size, num_classes, model_rate)
    model.apply(init_param)
    return model


if __name__ == '__main__':
    hidden_size = [64, 128, 256, 512, 512, 4096]
    model = vgg16(hidden_size, 10, 1)
    model_half = vgg16(hidden_size, 10, 0.5)
    for k, v in model_half.named_modules():
        print(k)
    x = torch.rand(5, 3, 224, 224)
    y = model(x)
    y1 = model_half(x)
    print(y.shape, y1.shape)

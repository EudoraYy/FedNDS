import argparse
import re

pattern = r'\d+\.\d+|\d+'
# python train_classifier.py --rounds 100 --lr 0.01 --mode aware --model_name resnet18 --dataset cifar10 --shardperuser 20 --group_name xxx --client_send_label --device cpu


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def get_base_params():
    parser = argparse.ArgumentParser(description='cfg')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--logname', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--frc', type=float)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--shardperuser', type=int)
    parser.add_argument('--device', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--partition_type', type=str, default='patho')
    # parser.add_argument('--probs', type=str)
    parser.add_argument('--inferen_batch', type=int)
    parser.add_argument('--select_mode', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--group_name', type=str)
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--client_send_label', action='store_true', default=False)
    parser.add_argument('--client_state', type=str, default='dynamic')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--scheduler_name', type=str, default='None')
    parser.add_argument('--milestones', type=int, nargs='+')    # Accept one or more space-separated values.
    parser.add_argument('--use_wandb', type=str_to_bool, default=True)
    parser.add_argument('--optimizer_name', type=str, default="SGD")
    parser.add_argument('--submodel_rate', type=float, nargs='+')
    parser.add_argument('--probs', type=float, nargs='+')
    parser.add_argument('--beta', type=float, default=0.5, help="historical dynamics weight")
    parser.add_argument('--init_v', type=float, default=9999.0, help="initial value of NDI")
    parser.add_argument('--fiarse_use_tcbgd', type=str_to_bool, default=True)
    parser.add_argument('--fiarse_aggregation', choices=['original', 'coverage'], default=None,
                        help='FIARSE CIFAR100/TinyImageNet only: original or initial-mask coverage averaging ablation')
    parser.add_argument('--par_alpha', type=float, default=0.3, help="dirichlet_cifar10 partition parameter")
    parser.add_argument('--weight_decay', type=float, default=5.0e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = get_base_params()
    print(args)

import copy
import torch
import swanlab
import torch.nn.functional as F
from collections import OrderedDict
from utils.config import cfg
from utils.utils import set_control, set_seed, get_model, make_scheduler, make_optimizer, generate_submodel_rate, \
    generate_fix_submodel_rate_list
from utils.utils import load_params_to_client_model, get_fiarse_active_ratio
from utils.fiarse_metrics import collect_fiarse_client_metrics, report_fiarse_round_metrics
from utils.data import get_transform, get_dataset, SplitDataset, non_iid, make_dataloader, get_inferen_data, \
    dataset_class_list, Dirichlet, wikitext2_contiguous_noniid, _make_contiguous_text_blocks


def main():
    set_control(cfg)
    set_seed(cfg['seed'])
    print('Config is OK!\n {}'.format(cfg))
    print_fiarse_aggregation(cfg)
    device = torch.device(cfg['device'])

    # use wandb to record
    if cfg['use_wandb']:
        swanlab.init(
            project=cfg['project_name'],
            name=cfg['run_name'],
            group=cfg['group_name'],
            config=cfg
        )
        print("Wandb is inited!")

    # global_dataset and non-iid progress
    transform = get_transform(cfg['dataset'], cfg['model_name'])
    dataset = get_dataset(cfg['dataset'], transform)
    print('Dataset {} is OK!'.format(cfg['dataset']))
    data_train_split, label_train_split = prepare_data_partitions(dataset)

    # global_model
    global_model = get_model(cfg['model_name'], dataset, model_rate=1)
    # optimizer
    optimizer = make_optimizer(global_model, cfg['lr'])
    # scheduler
    scheduler = make_scheduler(optimizer)

    # fix mode sub_rate list
    sub_model_rate_list = generate_fix_submodel_rate_list()
    # for grad-base mode we need konw class-list
    class_list = dataset_class_list(dataset['train'])
    # for grad-base pre-train the global model
    pre_run_train(cfg, global_model, class_list, dataset['train'], optimizer, device)
    # record last round users params and rate
    last_client_info = {i: [] for i in range(cfg['numbers'])}
    print('Experiment is beginning!')
    global_abs_updates = {}

    eta_g = 1
    largest_down_comm_cost = 0
    largest_server_flops = 0
    for i in range(cfg['rounds']):
        print(' ROUNDS {}'.format(i + 1), end=' ')
        num_activate_users = int(cfg['numbers'] * cfg['frc'])
        user_idxs = torch.randperm(cfg['numbers'])[:num_activate_users]
        clients_models_params = {}
        clients_models_shape = {}
        clients_models_abs_updates = {}
        clients_models_gradients = {}
        clients_models_masks = {}
        clients_models_threshold_masks = {}
        clients_models_update_masks = {}
        clients_fiarse_metrics = {}
        round_metrics = {}

        for client_order, user_idx in enumerate(user_idxs):
            # sub_dataset
            client_dataset = SplitDataset(dataset['train'], data_train_split[int(user_idx)])
            client_dataloader = make_dataloader(client_dataset, batch_size=cfg['batchsize'])
            label_mask = make_client_label_mask(dataset, label_train_split, user_idx, device)

            # sub_model
            rate = generate_submodel_rate(cfg['client_state'], sub_model_rate_list, user_idx)
            model_rate = 1 if cfg['mode'] == 'fiarse' else rate
            client_model = get_model(cfg['model_name'], dataset, model_rate=model_rate)

            # for GradFL sever don't know the client labels, so we should inferen its labels
            user_idx_label = get_client_dataset_label(user_idx, cfg, class_list, last_client_info, dataset,
                                                      label_train_split, device)
            # Prepare inference data for the current submodel-selection mode.
            inferen_data = get_inferen_data(cfg['mode'], cfg['inferen_batch'], class_list, user_idx_label,
                                            client_dataset, dataset['train'])

            # Select and load global parameters into the client model, recording indices and masks.
            load_params_to_client_model(global_model, client_model, inferen_data, cfg['mode'], device, rate,
                    cfg['select_mode'], label_mask, clients_models_shape, user_idx, i+1, global_abs_updates,
                    clients_models_masks, client_order)

            # Copy the initial FIARSE mask before local training; do nothing for other modes.
            save_client_threshold_mask(user_idx, clients_models_masks, clients_models_threshold_masks)

            # train model
            # Train the client and return absolute parameter changes and FIARSE update deltas.
            abs_updates, gradient = get_client_model_abs_updates(client_model, client_dataloader, label_mask, device,
                                    optimizer, clients_models_params, user_idx)

            # Record client updates, FIARSE masks and metrics, and the latest client model state.
            record_client_training_result(client_model, user_idx, rate, abs_updates, gradient,
                                          clients_models_threshold_masks, clients_models_update_masks,
                                          clients_fiarse_metrics, clients_models_masks, clients_models_abs_updates,
                                          clients_models_gradients, last_client_info)

        # Advance roll indices or update randGrad selection statistics before aggregation.
        global_abs_updates = update_round_algorithm_state(global_model, user_idxs, clients_models_shape,
            clients_models_abs_updates, global_abs_updates)

        # Aggregate client results using the selected algorithm; return round metrics and cost statistics.
        round_metrics, largest_down_comm_cost, largest_server_flops = aggregate_round_models(global_model, user_idxs,
             clients_models_shape, clients_models_params, clients_models_gradients, clients_models_masks,
             clients_models_threshold_masks, clients_models_update_masks, clients_fiarse_metrics, eta_g,
             largest_down_comm_cost, largest_server_flops)

        # Evaluate the aggregated global model and log test results with round metrics.
        glob_acc, glob_loss = run_global_test(global_model, dataset, device, round_metrics)
        scheduler.step()
    if cfg['use_wandb']:
        swanlab.finish()


def save_client_threshold_mask(user_idx, clients_models_masks, clients_models_threshold_masks):
    if cfg['mode'] == 'fiarse':
        clients_models_threshold_masks[int(user_idx)] = copy.deepcopy(clients_models_masks[int(user_idx)])


def record_client_training_result(client_model, user_idx, rate, abs_updates, gradient,
                                  clients_models_threshold_masks, clients_models_update_masks,
                                  clients_fiarse_metrics, clients_models_masks,
                                  clients_models_abs_updates, clients_models_gradients,
                                  last_client_info):
    if cfg['mode'] == 'fiarse':
        clients_models_update_masks[int(user_idx)] = get_fiarse_update_masks(gradient)
        clients_fiarse_metrics[int(user_idx)] = collect_fiarse_client_metrics(
            client_model, clients_models_threshold_masks[int(user_idx)],
            clients_models_update_masks[int(user_idx)])
        clients_models_masks[int(user_idx)] = get_fiarse_aggregation_mask(
            clients_models_threshold_masks[int(user_idx)],
            clients_models_update_masks[int(user_idx)],
            cfg['fiarse_aggregation_effective'],
            cfg.get('fiarse_use_tcbgd', True))
    clients_models_abs_updates[int(user_idx)] = copy.deepcopy(abs_updates)
    clients_models_gradients[int(user_idx)] = copy.deepcopy(gradient)
    last_client_info[int(user_idx)].clear()
    last_client_info[int(user_idx)].extend([rate, copy.deepcopy(client_model.state_dict())])


def update_round_algorithm_state(global_model, user_idxs, clients_models_shape,
                                 clients_models_abs_updates, global_abs_updates):
    if cfg['mode'] == 'roll':
        global_model.roll += 1
    if cfg['mode'] == 'randGrad':     # adding code
        global_abs_updates = combine_abs_updates_clients(global_model, user_idxs, clients_models_shape,
                                                         clients_models_abs_updates, global_abs_updates)
    return global_abs_updates


def aggregate_round_models(global_model, user_idxs, clients_models_shape,
                           clients_models_params, clients_models_gradients,
                           clients_models_masks, clients_models_threshold_masks,
                           clients_models_update_masks, clients_fiarse_metrics,
                           eta_g, largest_down_comm_cost, largest_server_flops):
    round_metrics = {}
    if cfg['mode'] == 'ecover':       # adding code
        combine_global_model_from_clients_ecover(global_model, user_idxs, clients_models_shape, clients_models_params, eta_g)
    elif cfg['mode'] == 'fiarse':
        threshold_ratios = [get_fiarse_active_ratio(clients_models_threshold_masks[int(user_idx)],
                        clients_models_shape[int(user_idx)].keys()) for user_idx in user_idxs]
        upload_ratios = [get_fiarse_active_ratio(clients_models_update_masks[int(user_idx)],
                        clients_models_shape[int(user_idx)].keys()) for user_idx in user_idxs]
        print('FIARSE threshold active ratio (thresholded tensors only, excludes BN): {:.4f}'.format(
            sum(threshold_ratios) / len(threshold_ratios)))
        print('FIARSE nonzero update ratio (thresholded tensors only, excludes BN): {:.4f}'.format(
            sum(upload_ratios) / len(upload_ratios)))
        round_metrics = report_fiarse_round_metrics(clients_fiarse_metrics)
        round_metrics['fiarse/coverage_aggregation_ablation'] = int(
            cfg['fiarse_aggregation_effective'] == 'coverage')
        largest_down_comm_cost, largest_server_flops = combine_global_model_from_clients_fiarse(global_model,
        user_idxs, clients_models_gradients, clients_models_masks, largest_down_comm_cost, largest_server_flops)
    else:
        combine_global_model_from_clients(global_model, user_idxs, clients_models_shape, clients_models_params)
    return round_metrics, largest_down_comm_cost, largest_server_flops


def print_fiarse_aggregation(cfg):
    if cfg['mode'] != 'fiarse':
        return

    effective = cfg['fiarse_aggregation_effective']
    description = (
        ' (ablation: average over initial-mask coverage, including zero updates)'
        if effective == 'coverage'
        else ' (existing implementation)'
    )
    print('FIARSE aggregation: requested={}, effective={}{}'.format(
        cfg.get('fiarse_aggregation', 'original'), effective, description))


def prepare_data_partitions(dataset):
    """Prepare partitions and metadata in the original order."""
    if cfg['dataset'] in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet']:
        if cfg['partition_type'] == 'patho':
            data_train_split, label_train_split = non_iid(dataset['train'], cfg['numbers'], cfg['shardperuser'])
        elif cfg['partition_type'] == 'dir':
            data_train_split, label_train_split = Dirichlet(dataset['train'], cfg['numbers'], cfg['par_alpha'])
        else:
            raise ValueError(f"Unsupported partition_type: {cfg['partition_type']}")
    elif cfg['dataset'] == 'wikitext2':
        cfg['vocab'] = dataset['train'].vocab
        cfg['num_tokens'] = len(dataset['train'].vocab)
        data_train_split, label_train_split = wikitext2_contiguous_noniid(dataset['train'], cfg['numbers'])
        _make_contiguous_text_blocks(dataset['test'], cfg['bptt'])
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")
    return data_train_split, label_train_split


def make_client_label_mask(dataset, label_train_split, user_idx, device):
    """Preserve text label splits and image class masks."""
    if cfg['dataset'] == 'wikitext2':
        return label_train_split[int(user_idx)]
    else:
        label_mask = torch.zeros(len(dataset['train'].classes), device=device)
        label_mask[label_train_split[user_idx]] = 1
    return label_mask


def compute_text_training_loss(client_model, batch, label_mask, device):
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch['label_split'] = label_mask
    output = client_model(batch)
    return output['loss']


def compute_image_training_loss(client_model, batch, label_mask, device):
    img, target = batch['img'], batch['label']
    img, target = img.to(device), target.to(device)
    output = client_model(img)
    if cfg['mode'] != 'fiarse':
        output = output.masked_fill(label_mask == 0, 0)
    return F.cross_entropy(output, target)


def get_client_model_abs_updates(client_model, client_dataloader, label_mask, device, optimizer,
                                 clients_models_params, user_idx):
    client_model_history = copy.deepcopy(client_model).to(device)
    abs_updates = OrderedDict()
    gradient = {}

    client_model.to(device)
    lr = optimizer.param_groups[0]['lr']
    local_optimizer = make_optimizer(client_model, lr)
    client_model.train()
    compute_training_loss = (compute_text_training_loss if cfg['dataset'] == 'wikitext2'
                             else compute_image_training_loss)

    for local_epoch in range(cfg['epoch']):
        for batch in client_dataloader:
            client_model.zero_grad()
            loss = compute_training_loss(client_model, batch, label_mask, device)
            loss.backward()
            if cfg['mode'] != 'fiarse':
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
            local_optimizer.step()
    client_model.cpu()
    client_model_history.cpu()

    for (name1, param1), (name2, param2) in zip(client_model.named_parameters(), client_model_history.named_parameters()):
        assert name1 == name2, "模型结构不一致！"
        abs_updates[name1] = torch.abs(param1.data - param2.data)

    if cfg['mode'] == 'fiarse':
        for (name, cur_param), (_, his_param) in \
            zip(client_model.state_dict().items(), client_model_history.state_dict().items()):
            gradient[name] = his_param - cur_param

    clients_models_params[int(user_idx)] = copy.deepcopy(client_model.state_dict())
    return copy.deepcopy(abs_updates), gradient


def get_fiarse_aggregation_mask(initial_masks, update_masks, aggregation_mode, use_tcbgd):
    """Choose aggregation support without changing model or upload masks.

    Coverage counts clients holding a coordinate even when its update is zero,
    matching randGrad's denominator convention. Original preserves the existing
    behavior, including its fixed-mask fallback when TCB-GD is disabled.
    """
    if aggregation_mode == 'coverage':
        return initial_masks
    if aggregation_mode == 'original':
        return update_masks if use_tcbgd else initial_masks
    raise ValueError('Unsupported FIARSE aggregation: {}'.format(aggregation_mode))


def get_fiarse_update_masks(gradients):
    return {
        name: torch.ne(grad, 0).detach().clone()
        for name, grad in gradients.items()
    }


def pre_run_train(cfg, model, class_list, dataset, optimizer, deivce):
    if cfg['mode'] == 'awareGrad' and cfg['is_pre_train']:
        model.to(deivce)
        lr = optimizer.param_groups[0]['lr']
        model_optimizer = make_optimizer(model, lr)

        shardperuser = cfg['shardperuser']
        inferen_batch = cfg['inferen_batch']
        public_per_class_num = inferen_batch // shardperuser

        public_dataset_label = []
        for i in range(len(class_list)):
            public_dataset_label.extend(class_list[i][: public_per_class_num])
        public_dataset = SplitDataset(dataset, public_dataset_label)
        public_dataset_dataloader = make_dataloader(public_dataset, public_per_class_num)

        pre_loss = []
        model.train()
        for local_epoch in range(cfg['pre_train_epoch']):
            local_loss = 0
            for batch in public_dataset_dataloader:
                img, target = batch['img'].to(deivce), batch['label'].to(deivce)
                model.zero_grad()
                output = model(img)
                loss = F.cross_entropy(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()
                local_loss += loss.data
            pre_loss.append(local_loss)
        print('Pre-train loss: {}'.format(pre_loss))
        model.cpu()
    else:
        print('mode {} don\'t need pre-train!'.format(cfg['mode']))


def combine_global_model_from_clients(global_model, user_idxs, clients_models_shape,
                                      clients_models_params):
    # Aggregate only client-provided values. Parameters not covered by any
    # selected submodel keep their previous global value.
    client_num_model_param = OrderedDict()
    client_param_sum = OrderedDict()
    global_params = copy.deepcopy(global_model.state_dict())

    for k, v in global_model.state_dict().items():
        client_num_model_param[k] = torch.zeros_like(v, dtype=torch.float32)
        client_param_sum[k] = torch.zeros_like(v)
    for user_idx in user_idxs:
        for k, v in clients_models_params[int(user_idx)].items():
            temp_shape = clients_models_shape[int(user_idx)][k]
            if k in client_param_sum:
                if v.dim() > 1:
                    client_param_sum[k][torch.meshgrid(temp_shape, indexing='ij')] += v
                    client_num_model_param[k][torch.meshgrid(temp_shape, indexing='ij')] += 1
                else:
                    client_param_sum[k][temp_shape] += v
                    client_num_model_param[k][temp_shape] += 1
            else:
                raise NameError('{} not in global_temp_params'.format(k))
    for k in global_params:
        observed_mask = client_num_model_param[k] > 0
        if observed_mask.any():
            averaged = client_param_sum[k][observed_mask] / client_num_model_param[k][observed_mask].to(client_param_sum[k].dtype)
            global_params[k][observed_mask] = averaged.to(global_params[k].dtype)
    global_model.load_state_dict(global_params)



def combine_global_model_from_clients_fiarse(global_model, user_idxs, clients_models_gradients,
                                             clients_models_masks, largest_down_comm_cost, largest_server_flops):
    tot_grad = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items()}
    tot_mask = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items()}
    for user_idx in user_idxs:
        for name, grad in clients_models_gradients[int(user_idx)].items():
            mask = clients_models_masks[int(user_idx)][name].to(dtype=grad.dtype, device=grad.device)
            tot_grad[name] += grad * mask
            tot_mask[name] += mask
    model_params = global_model.state_dict()
    for name in model_params.keys():
        # Update the parameters
        if torch.sum(torch.isnan(tot_grad[name])) > 0:
            print("NaN occurs. Terminate. ", flush=True)
            return None, None
        if ('running_mean' in name) or ('running_var' in name) \
                or ('num_batches_tracked' in name):
            # This is the normalization layer with tracking stats of inputs
            pass
        else:
            observed_mask = tot_mask[name] > 0
            avg_grad = torch.zeros_like(tot_grad[name])
            if observed_mask.any():
                avg_grad[observed_mask] = (
                    tot_grad[name][observed_mask] /
                    tot_mask[name][observed_mask].to(tot_grad[name].dtype)
                )
            avg_grad = torch.nan_to_num(avg_grad, nan=0.0,
                                        posinf=0.0, neginf=0.0)
            model_params[name] -= (1.0 * avg_grad)
    global_model.load_state_dict(model_params)

    flops_weighted_avg = sum(p.numel() for p in global_model.parameters())
    # Calculate model aggregation FLOPs and downstream communication.
    print("=" * 10 + "Aggregate_models and Downstream_communication" + "=" * 10)
    largest_down_comm_cost, largest_server_flops = calculate_flops_and_comm(user_idxs, clients_models_gradients,
           OrderedDict(), flops_weighted_avg, largest_down_comm_cost, largest_server_flops)
    return copy.deepcopy(largest_down_comm_cost), copy.deepcopy(largest_server_flops)


def combine_global_model_from_clients_ecover(global_model, user_idxs, clients_models_shape, clients_models_params,
                                             eta_g=0.9):
    # Aggregate client parameters into the global model.
    client_num_model_param = OrderedDict()
    global_params = OrderedDict()
    global_params_old = OrderedDict()
    param_delta_accumulator = OrderedDict()

    for k, v in global_model.state_dict().items():
        global_params[k] = v
        global_params_old[k] = v

    for k, v in global_model.state_dict().items():
        client_num_model_param[k] = torch.zeros_like(v)
        param_delta_accumulator[k] = torch.zeros_like(v)     # Parameter updates.
    for user_idx in user_idxs:
        for k, v in clients_models_params[int(user_idx)].items():
            temp_shape = clients_models_shape[int(user_idx)][k]
            if k in param_delta_accumulator:
                if v.dim() > 1:
                    param_delta_accumulator[k][torch.meshgrid(temp_shape, indexing='ij')] += \
                        (v - global_params_old[k][torch.meshgrid(temp_shape, indexing='ij')]) * len(user_idxs)
                    client_num_model_param[k][torch.meshgrid(temp_shape, indexing='ij')] += len(user_idxs)
                else:
                    param_delta_accumulator[k][temp_shape] += (v - global_params_old[k][temp_shape]) * len(user_idxs)
                    client_num_model_param[k][temp_shape] += len(user_idxs)
            else:
                raise NameError('{} not in global_temp_params'.format(k))
    # Normalize after accumulating updates from all clients.
    for k in param_delta_accumulator:
        # Avoid division by zero.
        mask = client_num_model_param[k] > 0
        if mask.any():  # Return False if all elements are False or the tensor is empty.
            param_delta_accumulator[k][mask] /= client_num_model_param[k][mask]
    # Update global parameters using the averaged values.
    for k, v in global_model.state_dict().items():
        global_params[k] += eta_g * param_delta_accumulator[k]
    global_model.load_state_dict(global_params)


def combine_abs_updates_clients(global_model, user_idxs, clients_models_shape,
                                clients_models_abs_updates, global_abs_updates):
    # Aggregate client updates into the global state.
    global_temp_abs_updates = OrderedDict()
    current_round_abs_updates = OrderedDict()
    selected_count = OrderedDict()

    for k, v in global_model.state_dict().items():
        # Initialize the sum of update magnitudes for this round.
        current_round_abs_updates[k] = torch.zeros_like(v, dtype=torch.float32)
        selected_count[k] = torch.zeros_like(v, dtype=torch.float32)
        # Initialize global average update magnitudes.
        if len(global_abs_updates) == 0:
            global_temp_abs_updates[k] = torch.full_like(v, fill_value=cfg['init_v'])
        else:
            global_temp_abs_updates[k] = copy.deepcopy(global_abs_updates[k])

    for user_idx in user_idxs:
        for k, v in clients_models_abs_updates[int(user_idx)].items():
            temp_shape = clients_models_shape[int(user_idx)][k]
            if k in current_round_abs_updates:
                if v.dim() > 1:
                    indices = torch.meshgrid(temp_shape, indexing='ij')
                    current_round_abs_updates[k][indices] += v
                    selected_count[k][indices] += 1
                else:
                    current_round_abs_updates[k][temp_shape] += v
                    selected_count[k][temp_shape] += 1
            else:
                raise NameError('{} not in global_temp_params'.format(k))

    # Compute average update magnitudes for this round.
    for k in current_round_abs_updates:
        observed_mask = selected_count[k] > 0
        current_avg = torch.zeros_like(current_round_abs_updates[k])
        current_avg[observed_mask] = current_round_abs_updates[k][observed_mask] / selected_count[k][observed_mask]
        current_round_abs_updates[k] = current_avg

    # 3. Weighted averaging stage.
    for k in global_temp_abs_updates:
        update_mask = selected_count[k] > 0
        non_update_mask = selected_count[k] == 0
        current_avg = current_round_abs_updates[k]

        # Keep the initial value when it is 9999 and the neuron is unobserved.
        mask_nonupdate = non_update_mask & (global_temp_abs_updates[k] == cfg['init_v'])
        global_temp_abs_updates[k][mask_nonupdate] = cfg['init_v']
        # Replace the initial value when it is 9999 and the neuron is observed.
        first_update_mask = update_mask & (global_temp_abs_updates[k] == cfg['init_v'])
        global_temp_abs_updates[k][first_update_mask] = current_avg[first_update_mask]
        # Retain historical values for unobserved neurons with values other than 9999; no explicit update is needed.
        # Apply weighted averaging to observed neurons with historical values other than 9999.
        non_first_update_mask = update_mask & (global_temp_abs_updates[k] != cfg['init_v'])
        global_temp_abs_updates[k][non_first_update_mask] = (
                cfg['beta'] * global_temp_abs_updates[k][non_first_update_mask] +
                (1 - cfg['beta']) * current_avg[non_first_update_mask]
        )
    # Calculate update-magnitude aggregation FLOPs and upstream communication.
    return copy.deepcopy(global_temp_abs_updates)


def calculate_flops_and_comm(user_idxs, clients_models_abs_updates, current_round_abs_updates, flops_weighted_avg,
                             largest_comm_cost, largest_server_flops):
    # Initialize statistics.
    flops_stats = {
        "total_flops (GFlops)": 0.0,
        "comm_cost(MB)": 0.0
    }
    # 2. Count parameter updates for each client.
    client_param_counts = []
    for user_idx in user_idxs:
        client_idx = int(user_idx)
        total_params = 0
        # Count all parameters updated by this client.
        for k, v in clients_models_abs_updates[client_idx].items():
            total_params += v.numel()  # Number of elements in this client parameter tensor.
        client_param_counts.append(total_params)
    # 3. Calculate FLOPs for accumulation.
    # Count one addition per updated parameter per client.
    flops_accumulate = sum(client_param_counts)  # Number of additions.
    # 4. Calculate FLOPs for averaging.
    # Count one division per accumulated parameter when averaging.
    # Count only parameters actually updated, rather than all global parameters.
    if len(current_round_abs_updates) == 0:
        flops_average = 0
    else:
        updated_params_count = 0
        for k in current_round_abs_updates:
            # Count parameters actually updated in this round.
            # Identify updated positions by nonzero entries in current_round_abs_updates[k].
            if torch.any(current_round_abs_updates[k] != 0):
                # Use the number of nonzero elements.
                updated_params_count += torch.count_nonzero(current_round_abs_updates[k] != 0).item()
            else:
                updated_params_count += 0
        flops_average = updated_params_count  # Number of divisions.
    flops_stats["total_flops (GFlops)"] += (flops_accumulate + flops_average + flops_weighted_avg) / 1e9

    bytes_per_param = 4  # torch.float32 uses four bytes per parameter.
    if len(current_round_abs_updates) == 0 and cfg['mode'] == 'aware':
        comm_mb = flops_weighted_avg * len(clients_models_abs_updates) * bytes_per_param / (1024 ** 2)
    else:
        comm_mb = sum(client_param_counts) * bytes_per_param / (1024 ** 2)
    flops_stats["comm_cost(MB)"] = comm_mb

    if largest_comm_cost < comm_mb:
        largest_comm_cost = comm_mb
    if largest_server_flops < flops_stats["total_flops (GFlops)"]:
        largest_server_flops = flops_stats["total_flops (GFlops)"]
    return copy.deepcopy(largest_comm_cost), copy.deepcopy(largest_server_flops)


def run_global_test(global_model, dataset, device, extra_metrics=None):
    # Evaluate the global model.
    test_dataloader = make_dataloader(dataset['test'], batch_size=32)
    correct, total = 0, 0
    global_model.to(device)
    global_model.eval()
    test_loss = torch.zeros((), device=device)
    with torch.no_grad():
        for batch in test_dataloader:
            if cfg['dataset'] == 'wikitext2':
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                output = global_model(batch)
                loss = output['loss']
                _, predicted = torch.max(output['score'], dim=1)
                target = batch['label'][:, 1:].contiguous()
                test_loss += loss.data * target.numel()
                total += target.numel()
                correct += (predicted == target).sum().item()
            else:
                img, target = batch['img'], batch['label']
                img, target = img.to(device), target.to(device)
                output = global_model(img)
                loss = F.cross_entropy(output, target)
                _, predicted = torch.max(output, dim=1)
                test_loss += loss.data * img.shape[0]
                total += img.shape[0]
                correct += (predicted == target).sum().item()
    glob_acc = correct / total
    glob_loss = test_loss / total
    if cfg['dataset'] == 'wikitext2':
        glob_ppl = torch.exp(torch.clamp(glob_loss, max=20)).item()
        print(' Global Model Token Acc: {}, Test Loss: {}, Perplexity: {}'.format(glob_acc, glob_loss, glob_ppl))
    else:
        glob_ppl = None
        print(' Global Model Acc: {}, Test Loss: {}'.format(glob_acc, glob_loss))
    if cfg['use_wandb']:
        log_data = {'acc': glob_acc, 'loss': glob_loss}
        if extra_metrics:
            log_data.update(extra_metrics)
        if glob_ppl is not None:
            log_data['ppl'] = glob_ppl
        swanlab.log(log_data)
    global_model.cpu()
    return glob_acc, glob_loss


def get_grad_client_dataset_label(user_id, cfg, class_list, last_client_info, dataset, device):
    model_name = cfg['model_name']
    numbers = cfg['numbers']
    shardperuser = cfg['shardperuser']
    inferen_batch = cfg['inferen_batch']
    public_per_class_num = inferen_batch // shardperuser
    if last_client_info[int(user_id)]:
        model_rate = last_client_info[int(user_id)][0]
        model_params = last_client_info[int(user_id)][1]
        model = get_model(model_name, dataset, model_rate)
        model.load_state_dict(model_params)
        model = model.to(device)
        user_class_acc = []
        label_list = []
        for i in range(len(class_list)):
            label_list.extend(class_list[i][:public_per_class_num])
        i_dataset = SplitDataset(dataset['train'], label_list)
        i_dataloader = make_dataloader(i_dataset, public_per_class_num, shuffle=False)
        with torch.no_grad():
            for batch in i_dataloader:
                correct_num = 0
                img, target = batch['img'].to(device), batch['label'].to(device)
                output = model(img)
                _, predicted = torch.max(output, dim=1)
                correct_num += (predicted == target).sum().item()
                user_class_acc.append(correct_num)
        user_idx_label = [i for i, _ in
                          sorted(enumerate(user_class_acc), key=lambda x: x[1], reverse=True)[: shardperuser]]
    else:
        user_idx_label = torch.randperm(len(class_list))[: shardperuser].tolist()
    return user_idx_label


def get_client_dataset_label(user_id, cfg, class_list, last_client_info, dataset, label_train_split, device):
    if cfg['dataset'] == 'wikitext2':
        return []
    if cfg['mode'] == 'awareGrad' and (not cfg['client_send_label']):
        return get_grad_client_dataset_label(user_id, cfg, class_list, last_client_info, dataset, device)
    elif cfg['mode'] == 'aware' or cfg['client_send_label']:
        return label_train_split[user_id]
    else:
        return []


if __name__ == "__main__":
    main()
"""FIARSE model budgets and logical upload payloads (no transport encoding)."""

import torch
from torch import nn


def collect_fiarse_client_metrics(model, initial_masks, update_masks):
    """Count retained parameter slots, including zero-valued BN/bias slots.

    Initial extraction masks define the model budget; nonzero local deltas
    define a separate, values-only sparse upload estimate. The implementation
    still stores dense tensors, and no indices/headers/serialization are counted.
    State buffers are excluded from model parameters and reported separately
    in the upload payload.
    """
    parameters = dict(model.named_parameters())
    bn_names = {
        '{}.{}'.format(module_name, name) if module_name else name
        for module_name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for name, _ in module.named_parameters(recurse=False)
    }
    metrics = dict.fromkeys([
        'dense_parameter_count', 'dense_parameter_bytes',
        'retained_parameter_count', 'retained_parameter_value_bytes',
        'retained_bn_parameter_count', 'upload_parameter_count',
        'upload_bn_parameter_count', 'upload_buffer_count',
        'upload_value_bytes', 'dense_update_value_bytes',
    ], 0)
    for name, parameter in parameters.items():
        retained = torch.count_nonzero(initial_masks[name]).item()
        updated = torch.count_nonzero(update_masks[name]).item()
        metrics['dense_parameter_count'] += parameter.numel()
        metrics['dense_parameter_bytes'] += parameter.numel() * parameter.element_size()
        metrics['retained_parameter_count'] += retained
        metrics['retained_parameter_value_bytes'] += retained * parameter.element_size()
        metrics['upload_parameter_count'] += updated
        if name in bn_names:
            metrics['retained_bn_parameter_count'] += retained
            metrics['upload_bn_parameter_count'] += updated

    for name, value in model.state_dict().items():
        updated = torch.count_nonzero(update_masks[name]).item()
        metrics['upload_value_bytes'] += updated * value.element_size()
        metrics['dense_update_value_bytes'] += value.numel() * value.element_size()
        if name not in parameters:
            metrics['upload_buffer_count'] += updated
    total = metrics['dense_parameter_count']
    metrics['retained_parameter_ratio'] = metrics['retained_parameter_count'] / total if total else 0.0
    metrics['upload_parameter_ratio'] = metrics['upload_parameter_count'] / total if total else 0.0
    return metrics


def report_fiarse_round_metrics(client_metrics):
    """Print per-client means/ranges and round upload totals; return log data."""
    rows = list(client_metrics.values())
    if not rows:
        return {}
    means = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    counts = [row['retained_parameter_count'] for row in rows]
    mib = 1024 ** 2
    print('FIARSE model budget incl. BN (initial mask, per-client mean): '
          '{:.1f}/{:.1f} parameters ({:.4%}); range [{}, {}]; BN {:.1f}; '
          'retained values {:.6f} MiB'.format(
              means['retained_parameter_count'], means['dense_parameter_count'],
              means['retained_parameter_ratio'], min(counts), max(counts),
              means['retained_bn_parameter_count'],
              means['retained_parameter_value_bytes'] / mib))
    print('FIARSE upload (nonzero deltas, per-client mean): '
          '{:.1f} parameters ({:.4%}), incl. BN {:.1f}; buffers {:.1f}; '
          'sparse values only {:.6f} MiB/client, {:.6f} MiB/round; '
          'dense update tensors {:.6f} MiB/round '
          '(indices/metadata/serialization excluded; no network transfer measured)'.format(
              means['upload_parameter_count'], means['upload_parameter_ratio'],
              means['upload_bn_parameter_count'], means['upload_buffer_count'],
              means['upload_value_bytes'] / mib,
              sum(row['upload_value_bytes'] for row in rows) / mib,
              sum(row['dense_update_value_bytes'] for row in rows) / mib))
    print('FIARSE dense parameter storage (per-client mean): {:.1f} parameters, '
          '{:.6f} MiB (parameter tensors only)'.format(
              means['dense_parameter_count'], means['dense_parameter_bytes'] / mib))
    log_data = {'fiarse/{}_mean'.format(key): value for key, value in means.items()}
    log_data.update({
        'fiarse/retained_parameter_count_min': min(counts),
        'fiarse/retained_parameter_count_max': max(counts),
        'fiarse/upload_value_bytes_round': sum(row['upload_value_bytes'] for row in rows),
        'fiarse/dense_update_value_bytes_round': sum(row['dense_update_value_bytes'] for row in rows),
    })
    return log_data

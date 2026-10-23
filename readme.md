## FedNDS: Neuron Dynamics-Aware Submodel Extraction for Heterogeneous Federated Learning **[Submitted to TMC'26]**

In this repository, we release the official code of FedNDS algorithm.

## Usage

Build and Activate a virtual environment
```
conda env create -f environment.yml
conda activate FedNDS
```

Prepare the four image datasets (MNIST, CIFAR10, CIFAR100, and TinyImageNet). This script does not prepare WikiText2.
```
cd FedNDS
bash scripts/init.sh
```

WikiText2 is downloaded and preprocessed automatically when a script in `scripts/wikitext2/` is first run with no existing dataset directories. The loader downloads the archive to `data/wikitext2/raw/`, extracts the token files, and saves the processed splits and vocabulary in `data/wikitext2/processed/`. The initial download requires network access; subsequent runs reuse the processed cache by default. No separate WikiText2 download command is needed.

For example, run from the repository root:

```bash
bash scripts/wikitext2/wikitext2_randGrad.sh
```

If the processed cache is absent but `data/wikitext2/raw/` already exists, preprocessing uses the existing raw files. The current loader checks directory existence, so an incomplete raw directory is not automatically downloaded again.


## FedNDS (Core code)
#### main.py
```
def get_client_model_abs_updates(client_model, client_dataloader, label_mask, device, optimizer, clients_models_params, user_idx)
combine_abs_updates_clients(global_model, user_idxs, clients_models_shape, clients_models_abs_updates, global_abs_updates)
```
#### models/resnet.py
```
def get_idx_rand_grad(self, rate, select_mode, gradient, epoch)
```
#### models/conv.py
```
def get_idx_rand_grad(self, rate, topmode, gradient, epoch)
```
#### models/transformer.py
```
def get_idx_rand_grad(self, rate, select_mode, gradient, epoch=None)
```

## FedNDS (DEMO)

Source: [scripts/ori/cifar10_randGrad.sh](scripts/ori/cifar10_randGrad.sh). This script runs CIFAR10 experiments with 4 and 2 shards per client.

```bash
for spu in 4 2
do
  echo "============= CIFAR10, mode: randGrad; partition: L_"$spu "============="
  starttime=`date +'%Y-%m-%d %H:%M:%S'`

  python main.py --mode randGrad \
                  --dataset cifar10 \
                  --model_name resnet18 \
                  --partition_type patho \
                  --shardperuser $spu \
                  --client_state real \
                  --select_mode l2 \
                  --rounds 1600 \
                  --epoch 1 \
                  --lr 0.01 \
                  --milestones 800 \
                  --scheduler_name MultiStepLR \
                  --project_name FedNDS-compare \
                  --run_name=FedNDS-cifar10-L$spu \
                  --group_name L_$spu \
                  --device 0

  endtime=`date +'%Y-%m-%d %H:%M:%S'`     # Save the time in a human-readable format.
  start_seconds=$(date --date="$starttime" +%s);
  end_seconds=$(date --date="$endtime" +%s);      # Convert to a Unix timestamp to calculate script runtime.
  echo 'running time is:' $((end_seconds-start_seconds)) 's'
done
```

Run the same script from the repository root:

```bash
bash scripts/ori/cifar10_randGrad.sh
```

## Baseline implementations and reproducibility

This section reports the implementation mechanisms and configurations verifiable from this release. Launch scripts establish intended configurations; they do not establish completed experiments or validation-based hyperparameter selection.

### Implementation source and method-specific mechanisms

All baselines are implemented in the common framework in [main.py](main.py), with parameter extraction and mode dispatch in [utils/utils.py](utils/utils.py). Architecture-specific selection is implemented in [models/conv.py](models/conv.py), [models/resnet.py](models/resnet.py), and [models/transformer.py](models/transformer.py). This repository acknowledges adaptation from the official code of [FedDSE](https://github.com/yaboojia/GradFL). Per-method upstream commits and code provenance are not recorded in the supplied files. Therefore, these implementations are described here as local implementations in an adapted common framework, without claiming that each baseline executes its official code unchanged.

| Method | Mode | Reproduced mechanism | Method-specific configuration and implementation choices |
|---|---|---|---|
| HeteroFL | `hetero` | Nested submodels formed from the first channels of each layer through `get_idx_hetero`. | Uses the common capacity process and optimizer. `select_mode` is unused in this branch, even where a script supplies it. |
| FedRolex | `roll` | Cyclic channel windows through `get_idx_roll`. | Offset is `global_model.roll + client_order`; the global counter advances once per round. This client-dependent within-round scheduling is the precise local implementation convention. |
| FedDSE | `aware` | A forward pass of the global model on a local client batch provides activation scores for channel selection through `get_idx_aware`. | Scripts use `select_mode=absmax`. The inference batch defaults to 128 and is capped by local dataset size. This mode does not activate `awareGrad` pretraining. |
| FIARSE | `fiarse` | Layer-wise magnitude thresholding, sparse parameter masks, and threshold-controlled biased gradients (TCB-GD). | Uses full-shape masked networks in [models/masked_conv.py](models/masked_conv.py), [models/masked_resnet.py](models/masked_resnet.py), and the masked Transformer components. TCB-GD is enabled by default. All supplied FIARSE launch scripts explicitly set both momentum and weight decay to zero. Channel rate `r` is converted to parameter retention `r²`, an approximate budget alignment rather than exact equality of parameters or FLOPs. |
| FedEcover | `ecover` | Persistent per-layer pools of unused indices; sampling and replenishment encourage coverage through `get_idx_ecover`. | Uses a dedicated delta aggregator. The actual server step size is `eta_g=1` in `main.py`, overriding the aggregation function's default 0.9. `select_mode` does not control this branch. |
| FedNDS (ours) | `randGrad` | Random selection in round 1, followed by selection using historical absolute local parameter changes aggregated into neuron dynamics statistics. | Scripts use `select_mode=l2`; parser defaults are `beta=0.5` and `init_v=9999.0`. These are FedNDS settings, not additional baseline hyperparameters. |

The common submodel aggregator averages each coordinate over clients containing it and preserves uncovered global coordinates. Client optimizers are recreated for each invocation, without persistent optimizer state across rounds. Non-FIARSE local training clips gradient norms to 1.

FIARSE defaults to `fiarse_aggregation=original`. With TCB-GD enabled, its current implementation averages deltas over clients with nonzero updates at each coordinate; with TCB-GD disabled, it uses initial masks. `fiarse_aggregation=coverage` changes the denominator to initial-mask coverage.

For WikiText2, channel-selection mechanisms are adapted to embedding dimensions within attention heads and feed-forward hidden dimensions, while retaining the vocabulary output space and propagating compatible indices. These are architecture-specific adaptations.

### Configuration resolution and shared hyperparameters

[utils/config.yml](utils/config.yml) supplies base values. [utils/globals.py](utils/globals.py) defines CLI arguments. `set_control` overwrites YAML entries with every non-`None` parsed value, including parser defaults for flags not explicitly supplied.

| Parameter | Supplied experiment settings unless explicitly overridden |
|---|---|
| Seed | 2025; no active multi-seed sweep is supplied |
| Clients | 100; uniform sampling without replacement each round |
| Participation | Images: 0.1 (10 clients); WikiText2: 0.2 (20 clients) |
| Rounds | 1600 |
| Local epochs | Images: 1; WikiText2: 2 |
| Batch size | 16 |
| Optimizer | SGD, momentum 0.9, weight decay `5e-4`; FIARSE exceptions below |
| Learning rate | 0.01 |
| Scheduler | Non-FIARSE scripts: MultiStepLR, milestone 800, gamma 0.1; one step per global round. FIARSE scripts omit the scheduler flag and use the parser default `None`, so the learning rate does not decay during the 1600-round runs. |
| Capacity candidates | `[1, 0.5, 0.25, 0.125, 0.0625]`, each with probability 0.2 |
| Capacity process | Scripts set `client_state=real`; perturbation `alpha=0.0625` |
| FedDSE inference batch | 128, capped by local dataset size |

FIARSE scripts explicitly use momentum 0 and weight decay 0 to keep consistent with the configurations in FIARSE's original paper.

In `real` mode, a base capacity is resampled at each invocation. For interior base rates, `[base+alpha, base, base-alpha]` has probabilities `[0.01, 0.98, 0.01]`. At base 1, probabilities are `[0, 0.01, 0.99]`; at the minimum base `alpha`, they are `[0, 1, 0]`. This simulates stochastic capacity rather than measured device traces. `fix` assigns one sampled capacity per client; `dynamic` resamples unperturbed candidates.
### Experiment scripts and dataset configurations

| Script group | Dataset / architecture | Methods and partition settings |
|---|---|---|
| `scripts/init.sh` | Dataset preparation | Downloads/prepares datasets; not a training experiment |
| `scripts/ori/mnist_*.sh` | MNIST / Conv | FedDSE, FedEcover, HeteroFL, FedRolex, FedNDS; shards: 4, 2 |
| `scripts/ori/cifar10_*.sh` | CIFAR10 / ResNet18 | The same five methods; shards: 4, 2 |
| `scripts/ori/cifar100_*.sh` | CIFAR100 / ResNet18 | The same five methods; shards: 10, 20 |
| `scripts/ori/tinyimagenet_*.sh` | TinyImageNet / ResNet34 | The same five methods; shards: 20, 40 |
| `scripts/fiarse/*_fiarse.sh` | MNIST / masked Conv; CIFAR10 and CIFAR100 / masked ResNet18; TinyImageNet / masked ResNet34 | FIARSE; shards respectively: 4, 2; 4, 2; 10, 20; 20, 40 |
| `scripts/dirichlet_cifar10` | CIFAR10 / ResNet18, masked for FIARSE | FedDSE, FedEcover, FIARSE, HeteroFL, FedRolex, FedNDS; Dirichlet label partition with concentration 0.3 |
| `scripts/wikitext2` | WikiText2 / Transformer | The same six methods; contiguous corpus partition (non-iid); LR 0.01 |

WikiText2 is split into consecutive blocks of 64 tokens, with each client receiving a consecutive block range. This is contiguous text non-IID partitioning, despite historical echo messages saying “shard-based”; image shard/Dirichlet settings do not control this branch. Transformer defaults are embedding size 256, 8 heads, feed-forward size 512, 4 layers, dropout 0.2.
## Acknowledgements
Some parts of our code and implementation has been adapted from [FedDSE](https://github.com/yaboojia/GradFL) repository.
# DiSI: Distributional-Specific Indistinguishability for Trustworthy Machine Learning

This repository implements DiSI (Distributional-Specific Indistinguishability), a framework for provable trustworthy machine learning. DiSI provides robust gradient aggregation mechanisms to defend against data poisoning, backdoor attacks, and memorization across multiple domains including LLM fine-tuning, image classification, and diffusion models.

## Repository Structure

```
DiSI/
├── main.py                          # Entry point for LLM memorization task (distributed)
├── train_DiSI_llm_memorization.py   # LLM memorization defense training
├── train_DiSI_llm_backdoor.py       # LLM backdoor defense training
├── train_DiSI_image_backdoor.py     # Image backdoor defense training
├── train_DiSI_diffusion.py          # Diffusion model fine-tuning with DiSI
├── evaluate_DiSI_llm_backdoor.py    # Evaluation for LLM backdoor (ASR, CA, PPL)
├── config.py                        # Configuration for diffusion training
├── dataset.py                       # Dataset class for diffusion training
├── ddoptimizer.py                   # Data-dependent optimizer
├── indexeddataset.py                # Indexed/tagged dataset wrappers
├── parser_config.py                 # Argument parser for distributed training
├── utils.py                         # Shared utilities (communication, aggregation, logging)
├── image_attacks.py                 # Image backdoor attack implementations (SSBA, SIG, LF, Trojan, BadNet, etc.)
├── run_DiSI_llm_backdoor.sh         # Script to run all LLM backdoor experiments
├── run_DiSI_llm_memorization.sh     # Script to run LLM memorization experiments
├── run_DiSI_image_backdoor.sh       # Script to run image backdoor experiments
├── configs/                         # YAML configs for LLM backdoor training
│   ├── jailbreak/                   #   Jailbreak task configs (badnet, sleeper, vpi, mtba, ctba)
│   ├── refusal/                     #   Refusal task configs (badnet, sleeper, vpi, mtba, ctba)
│   └── deepspeed/                   #   DeepSpeed configs
├── data/                            # Datasets and test data
│   ├── poison_data/                 #   Poisoned training data (jailbreak & refusal)
│   ├── test_data/                   #   Clean and poisoned test data
│   ├── diffusion_images/            #   Images for diffusion fine-tuning
│   ├── lavita/                      #   ChatDoctor-HealthCareMagic dataset
│   ├── SSBA_cifar_pt/               #   SSBA attack data
│   ├── LF_cifar_npy/                #   Label-Flipping attack data
│   └── Trojan_cifar_npz/            #   Trojan attack data
├── datasets/                        # Model and dataset definitions (CIFAR-10, ResNets, WRN, VIT)
├── llamafactory/                    # Modified LlamaFactory library for LLM training
└── poison_tools/                    # Poison dataset generation utilities
```

## Tasks

DiSI supports four main experimental tasks:

| Task | Script | Description |
|------|--------|-------------|
| **LLM Memorization** | `run_DiSI_llm_memorization.sh` | Defense against canary memorization in LLM fine-tuning |
| **LLM Backdoor** | `run_DiSI_llm_backdoor.sh` | Defense against backdoor attacks on LLMs (jailbreak & refusal) |
| **Image Backdoor** | `run_DiSI_image_backdoor.sh` | Defense against image classification backdoor attacks |
| **Diffusion** | `train_DiSI_diffusion.py` | Fine-tuning Stable Diffusion with DiSI aggregation |

## Setting Up Environments

Due to differing dependency requirements, two separate environments are needed.

### Environment 1: All Tasks Except Diffusion

`requirements_1.txt` covers LLM memorization, LLM backdoor, and image backdoor tasks.

```bash
conda create --name disi python=3.10
conda activate disi
pip install -r requirements_1.txt
```

### Environment 2: Diffusion Task

`requirements_2.txt` covers the diffusion fine-tuning task.

```bash
conda create --name disi_diffusion python=3.10
conda activate disi_diffusion
pip install -r requirements_2.txt
```

### Hugging Face Authentication

For tasks that download models from Hugging Face (e.g., LLaMA-2, Stable Diffusion), log in first:

```bash
huggingface-cli login
```

## Reproducing Results

Due to the high computational requirements, all training algorithms are implemented using CUDA. 

### LLM Memorization Task

This task trains a language model (e.g., OPT-125M, GPT-2) with DiSI defense against canary memorization attacks, using distributed multi-GPU training.

```bash
conda activate disi
bash ./run_DiSI_llm_memorization.sh
```

Key parameters (editable in the script):
- `model_name`: Base model (`opt-125m`, `gpt2`, `gpt2-medium`)
- `defense`: Aggregation defense (`none`, `majority`, `topk`)
- `gpu_per_node`: Number of GPUs (default: 2)
- `batch_size`: Batch size (default: 300)
- `canary_repeat`: Number of canary repetitions (default: 5)

### LLM Backdoor Task

This task trains LLaMA-2-7B-Chat with LoRA under federated-style grouped sampling, defending against various backdoor attacks using robust gradient aggregation.

```bash
conda activate disi
bash ./run_DiSI_llm_backdoor.sh           # Run all 30 experiments
bash ./run_DiSI_llm_backdoor.sh --dry-run # Preview commands without running
```

The script runs all combinations of:
- **Aggregation methods**: `mean`, `majority_vote`, `topk`
- **Tasks**: `jailbreak`, `refusal`
- **Attack methods**: `badnet`, `sleeper`, `vpi`, `mtba`, `ctba`

To run a single experiment manually:

```bash
python train_DiSI_llm_backdoor.py configs/jailbreak/llama2_7b_chat/llama2_7b_jailbreak_badnet_lora_federated.yaml \
    --aggregation topk \
    --num_benign_groups 18 \
    --num_backdoor_groups 2
```

After training, evaluate the model:

```bash
python evaluate_DiSI_llm_backdoor.py
```

The evaluation script computes:
- **ASR** (Attack Success Rate) on poisoned test data
- **CA** (Clean Accuracy) on clean test data
- **PPL** (Perplexity) and loss metrics

### Image Backdoor Task

This task trains image classifiers (ResNet, ViT) on CIFAR-10/CIFAR-100 with DiSI defense against various image backdoor attacks.

```bash
conda activate disi
bash ./run_DiSI_image_backdoor.sh
```

Supported attacks: `SSBA`, `SIG`, `LF`, `Trojan`, `AdapBlend`, `Blended`, `BadNet`

Supported defenses: `majority`, `topk`, `mean`

To customize:

```bash
python train_DiSI_image_backdoor.py \
    --expected_batchsize 500 \
    --EPOCH 10 \
    --lr 0.1 \
    --attack AdapBlend \
    --dataset cifar100 \
    --model vit \
    --defense topk
```

### Diffusion Task

This task fine-tunes Stable Diffusion v1.4 with LoRA and DiSI's per-sample gradient aggregation on custom image data.

```bash
conda activate disi_diffusion
python train_DiSI_diffusion.py
```

Configuration is managed in `config.py`:
- `MODEL_ID`: Base diffusion model (default: `CompVis/stable-diffusion-v1-4`)
- `ENABLE_PSG`: Enable per-sample gradient aggregation (default: `True`)
- `ENABLE_GAP_OPTIMIZER`: Enable gap-based optimizer (default: `True`)
- `ENABLE_LORA`: Use LoRA fine-tuning (default: `True`)
- `LORA_RANK` / `LORA_ALPHA`: LoRA hyperparameters (default: 64 / 128)
- `BATCH_SIZE`: Training batch size (default: 32, tuned for A100 80GB)
- `MAX_STEPS`: Total training steps (default: 2000)

Place training images in `./data/diffusion_images/` with naming format `ClassID_ImageID.jpg`.

## Aggregation Methods

DiSI implements several robust gradient aggregation strategies:

| Method | Description |
|--------|-------------|
| `mean` | Standard coordinate-wise averaging (baseline) |
| `majority_vote` | Mode-based voting with gap detection |
| `topk` | Gap-count based selection with quantization |

## Clarification of Arguments

### Distributed Training Arguments (`parser_config.py`)
- **`--master-addr`**: IP address of the master node. Default: `localhost` (single machine). For multi-node, set to the master node's address and assign a unique `--rank-node` to each node.
- **`--master-port`**: Port for inter-process communication. Default: `29500`.
- **`--rank-node`**: Rank of current node (machine).
- **`--gpu-per-node`**: Number of GPUs per node. Uses `cuda:0` through `cuda:gpu_per_node-1`.
- **`--num-nodes`**: Number of physical nodes. Default: `1`.

### LLM Training Arguments (`parser_config.py`)
- **`--task`**: Task type. Choices: `cifar10`, `diffusion`, `llm`, `backdoor`.
- **`--epochs`**: Number of training epochs.
- **`--batch-size`**: Batch size (subgroup size when `--full-mode` is enabled).
- **`--physical-size`**: Physical batch size for memory management. Set to `1` for LLM tasks.
- **`--lr`**: Learning rate. Set to `1` for `--full-mode`.
- **`--mgn`**: Maximum gradient norm for update clipping. Set large to disable.
- **`--num-local-iter`**: Number of local iterations. Automatically enables `--full-mode` when > 1.
- **`--defense`**: Aggregation defense method (`none`, `majority`, `topk`).
- **`--eps`**: Privacy guarantee parameter. `0` = no noise.
- **`-r`**: Resume from checkpoint in `--save-dir`.
- **`--save-dir`**: Directory for logs and checkpoints.
- **`--model-name-or-path`**: Base model name (e.g., `gpt2`, `opt-125m`, `gpt2-medium`).
- **`--block-size`**: Token length of each data point. Default: `1024`.
- **`--canary-repeat`**: Number of canary repetitions.
- **`--add-canary`**: Add canary to the training set.
- **`--canary-prompt`**: Prompt style for canary (`normal`, `wizard`).
- **`--wiki-size`**: Size of the wiki-103 subset.
- **`--do-ref-model`**: Use a reference model during evaluation.

### Image Backdoor Arguments (`utils.py`)
- **`--expected_batchsize`**: Expected batch size (required). Default: `1000`.
- **`--EPOCH`**: Number of training epochs (required). Default: `50`.
- **`--lr`**: Learning rate (required). Default: `0.01`.
- **`--log_dir`**: Log directory (required). Default: `logs`.
- **`--beta`**: Momentum beta. Default: `0.9`.
- **`--attack`**: Backdoor attack type. Choices: `BadNet`, `Blended`, `AdapBlend`, `Trojan`, `SIG`, `LF`, `SSBA`. Default: `SSBA`.
- **`--dataset`**: Dataset. Choices: `cifar10`, `cifar100`. Default: `cifar100`.
- **`--model`**: Model architecture. Choices: `resnet`, `vit`. Default: `vit`.
- **`--source_num`**: Number of source groups. Default: `10`.
- **`--poison_rate`**: Poisoning rate. Default: `0.1`.
- **`--target_label`**: Target label for the backdoor attack. Default: `0`.
- **`--accum_num`**: Gradient accumulation steps. Default: `10`.
- **`--accum_enabled`**: Enable gradient accumulation. Default: `False`.
- **`--defense`**: Defense method. Choices: `none`, `majority`, `topk`. Default: `none`.
- **`--use_abl_loss`**: Use ABL (Anti-Backdoor Learning) defense loss. Default: `False`.
- **`--abl_drop_ratio`**: Drop ratio for ABL defense. Default: `0.2`.

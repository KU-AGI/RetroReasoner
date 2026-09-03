<h1 align="center">RetroReasoner</h1>

<p align="center">
  A Reasoning LLM for Strategic Retrosynthesis Prediction
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.12666">📄 Paper</a> •
  <a href="https://huggingface.co/KU-AGI/RetroReasoner-RL">🤗 RetroReasoner (RL)</a> •
  <a href="https://huggingface.co/KU-AGI/RetroReasoner-RoundTrip-8B">🤗 Round-Trip Model</a> •
  <a href="https://huggingface.co/datasets/KU-AGI/RetroReasoner-data">🤗 Dataset</a>
</p>

---

RetroReasoner is a reasoning LLM for single-step retrosynthesis prediction:
given a target product, it generates a disconnection-based rationale before
proposing reactants, trained with supervised fine-tuning followed by
reinforcement learning against a round-trip (forward-synthesis) reward. This
repository contains the evaluation code for reproducing the **main
in-distribution results** reported for **RetroReasoner (RL)** in the paper
(Findings of EMNLP 2026).

> **Release scope.** Only the RL-trained model and the main in-distribution
> test set are currently released — the SFT checkpoint, training code,
> ablations, and out-of-distribution splits are not (yet) included here.

## Contents

- [Model Zoo](#model-zoo)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Environment variables](#environment-variables)
- [Repository structure](#repository-structure)
- [Citation](#citation)

## Model Zoo

| Model | Description | Checkpoint |
|---|---|---|
| RetroReasoner (RL) | SFT + RL retrosynthesis model evaluated here | [KU-AGI/RetroReasoner-RL](https://huggingface.co/KU-AGI/RetroReasoner-RL) |
| Round-Trip (8B) | Forward-synthesis model used to score reactant feasibility and as the RL reward model | [KU-AGI/RetroReasoner-RoundTrip-8B](https://huggingface.co/KU-AGI/RetroReasoner-RoundTrip-8B) |

Both are fine-tuned from Qwen3-8B.

## Dataset

`main-evaluation.json` — 500 reactions held out from
[ORDerly](https://github.com/sustainable-processes/orderly) for the main in-distribution
retrosynthesis test set. Hosted at
[`KU-AGI/RetroReasoner-data`](https://huggingface.co/datasets/KU-AGI/RetroReasoner-data)
(`testset/main-evaluation.json`) on the Hub.

## Setup

```bash
pip install -r requirements.txt
```

Template counting in `eval_2_save_metrics.py` uses the vendored
`localmapper/` package (a modified fork of
[localmapper](https://pypi.org/project/localmapper/) that returns
atom-mapping templates as a dict rather than a flat string — kept in this
repo rather than the PyPI package). It depends on `torch`/`dgl`/`dgllife`,
installed separately from `requirements.txt`:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps
pip install dgllife
```

A plain `pip install torch dgl` will resolve a `torch`/`dgl` pair that
doesn't work together — DGL's official wheels lag behind PyTorch releases
(the latest available is built against torch 2.4–2.6), so an unpinned
install pulls in a newer `torch` with no matching `dgl` binary. The versions
above are pinned instead: `torch==2.7.1` (cu128) for GPU architectures newer
than what DGL's own wheel index targets (e.g. Blackwell/B200 — you'll need
this if `torch.cuda.get_arch_list()` doesn't include your GPU's compute
capability on an older torch), paired with `dgl` from DGL's `torch-2.4/cu124`
wheel index, which works fine at runtime for `localmapper`'s usage despite
the version mismatch in its own metadata (pip will print a stale dependency
warning here — safe to ignore). If your GPU is fully covered by DGL's own
wheel index, you can instead follow
https://www.dgl.ai/pages/start.html directly for a version-matched pair.

## Usage

### 1. Serve the models with vLLM

```bash
# RetroReasoner (RL), e.g. 4 GPUs starting at port 8000
bash scripts/serve_vllm.sh retroreasoner 4 8000
```

Run steps 2 and 3 below, then stop the RetroReasoner (RL) servers and serve
the round-trip model, **leaving GPU 0 free** (see the GPU 0 caveat below):

```bash
# Round-trip model on GPUs 1,2,3, starting at port 8091
bash scripts/serve_vllm.sh roundtrip 1,2,3 8091
```

Each invocation launches one `vllm serve` process per GPU
(`CUDA_VISIBLE_DEVICES=$GPU`), logging to `logs/`. `<GPUS>` accepts either a
count (GPUs `0..count-1`) or an explicit comma-separated list; adjust to your
hardware, and point `VLLM_BASE_URLS`/`ROUNDTRIP_BASE_URLS` at the port(s) you
actually started (`START_PORT + position in the list`).

> **GPU 0 caveat.** `eval_2_save_metrics.py` also runs `localmapper` locally
> (for template counting), which binds to physical GPU 0 by default
> regardless of `CUDA_VISIBLE_DEVICES` scoping elsewhere in the process. If a
> round-trip vLLM replica also lands on GPU 0, the two will contend for that
> GPU's memory — `localmapper` runs 30 concurrent worker processes, and
> failures under contention are silently swallowed, undercounting Template
> Diversity without any visible error (we hit this ourselves: naively serving
> the round-trip model on GPUs 0–3 dropped Template Diversity from ~3.1 to
> well under 1). Either leave GPU 0 out of your round-trip serving GPUs (as
> above), or, if you have a GPU free elsewhere, set `LOCALMAPPER_DEVICE` (see
> below) to point `localmapper` at it instead and use all of GPUs 0–3 for
> round-trip serving.

### 2. Generate predictions

```bash
VLLM_BASE_URLS=localhost:8000,localhost:8001,localhost:8002,localhost:8003 \
  python eval_1_save_outputs.py
```

Downloads `main-evaluation.json` from the Hub (or reads it from
`RETRO_TEST_DATA_PATH` if set), queries the served model at `temperature=0.0`
(greedy, Exact@1) and `temperature=1.2` (100 samples, Exact@100), and writes
results to `outputs/retro_test/RetroReasoner(RL)_temp{0.0,1.2}.json`.

### 3. Compute metrics

```bash
ROUNDTRIP_BASE_URLS=localhost:8091,localhost:8092,localhost:8093 \
  python eval_2_save_metrics.py
```

Round-trips each predicted set of reactants through the round-trip model,
computes `Exact@1`, `Round-trip@1`, `Exact@100`, `Round-trip@100`, `Feasible
Ratio`, and `Template Diversity`, writes them back into the same
`outputs/retro_test/*.json` files, and prints a final summary.

vLLM's sampling is not fully deterministic across serving sessions, so exact
decimal values (particularly `Exact@100`, `Round-trip@100`, and `Template
Diversity`, which depend on `temperature=1.2` sampling) will vary slightly
run to run — expect small differences at the second or third decimal place
rather than an exact match. If Template Diversity comes out much lower than
the rest (e.g. ~1.7 instead of ~3.1), see the **GPU 0 caveat** above — a
round-trip vLLM replica sharing GPU 0 with `localmapper` silently undercounts
templates without raising an error.

## Environment variables

| Variable | Used by | Default | Description |
|---|---|---|---|
| `VLLM_MODEL_NAME` | eval_1 | `KU-AGI/RetroReasoner-RL` | Model name registered with vLLM |
| `VLLM_BASE_URLS` | eval_1 | `localhost:8000` | Comma-separated `host:port` list to load-balance across |
| `ROUNDTRIP_MODEL_NAME` | eval_2 | `KU-AGI/RetroReasoner-RoundTrip-8B` | Round-trip model name registered with vLLM |
| `ROUNDTRIP_BASE_URLS` | eval_2 | `localhost:8090` | Comma-separated `host:port` list for the round-trip servers |
| `ROUNDTRIP_PARALLELISM_PER_INSTANCE` | eval_2 | `min(3, num servers)` | Max concurrent round-trip calls per test instance |
| `RETRO_TEST_DATA_PATH` | eval_1, eval_2 | unset (downloads from HF Hub) | Local override path to `main-evaluation.json` |
| `HF_DATASET_REPO_ID` | eval_1, eval_2 | `KU-AGI/RetroReasoner-data` | HF dataset repo id to download the test set from |
| `HF_DATASET_FILENAME` | eval_1, eval_2 | `testset/main-evaluation.json` | Path within the dataset repo |
| `LOCALMAPPER_DEVICE` | eval_2 | `cuda` (physical GPU 0) | Device for `localmapper`'s atom-mapping model — see the GPU 0 caveat above |

## Repository structure

```
RetroReasoner/
├── eval_1_save_outputs.py   # query the served model, save raw predictions
├── eval_2_save_metrics.py   # round-trip + score the saved predictions
├── evaluator/
│   └── smiles_evaluator.py   # exact-match / fingerprint-similarity metrics
├── localmapper/              # vendored fork, used for reaction template counting
├── scripts/
│   └── serve_vllm.sh         # vLLM launch helper
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{ko2026retroreasoner,
  title     = {RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction},
  author    = {Ko, Hanbum and Lee, Chanhui and Kim, Ye Rin and Hormazabal, Rodrigo and Han, Sehui and Lim, Sungbin and Kim, Sungwoong},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

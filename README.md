# FedOSSA: Noise Calibrated Spectral Aggregation with Secure Aggregation for Federated Fine Tuning of LLMs

Code for federated LoRA fine tuning of large language models with **FedOSSA**
(Federated Optimal Spectral Shrinkage Aggregation) and its secure aggregation variant
**FedOSSA+SecAgg**, together with the baselines used in the paper.

FedOSSA aggregates the clients' LoRA updates in the exact product space
G = Σₖ pₖ BₖAₖ, applies the Frobenius optimal singular value shrinker of Gavish and
Donoho calibrated to the aggregation noise floor ν = σ_E √(Σₖ pₖ²), and refactors the
result to the adapter rank r. Because it only needs the weighted **sum** of the client
updates, it composes with pairwise masking secure aggregation without any loss of utility.

## Methods

| Key | Method | Aggregation |
|---|---|---|
| `centralized` | Centralized | pooled data, no federation (reference) |
| `fedavg` | FedAvg / FedIT | LoRA factors A and B averaged separately |
| `fedit` | FedIT | identical to `fedavg` on LoRA (off by default; add it to run it under its own name) |
| `fedptuning` | FedPTuning | federated prompt tuning, 20 soft prompt tokens averaged |
| `fedask` | FedASK | randomized rank r sketch of the product space aggregate |
| `fedossa` | **FedOSSA** | product space aggregate → optimal shrinkage at the noise floor → rank r refactor |
| `fedossa_secagg` | **FedOSSA+SecAgg** | FedOSSA under pairwise masking secure aggregation |

LLMs: `gpt2-l` (GPT-2 Large), `bloom-1b1` (BLOOM-1.1B), `qwen3-1.7b` (Qwen3-1.7B).
Datasets: `dolly` (Databricks Dolly 15k), `alpaca` (Alpaca cleaned), `natinst`
(Super-NaturalInstructions). Models and datasets are downloaded automatically from the
Hugging Face Hub on first use.

## Repository layout

| File | Purpose |
|---|---|
| `fedossa_core.py` | Shared engine: data splits, models, all methods, trainers, and the three studies |
| `run_sweep.py` | Main comparison: test perplexity, CIDEr-D, training loss per round |
| `run_dp.py` | Robustness to aggregation noise: test perplexity versus ρ |
| `run_async.py` | Robustness to stragglers: test perplexity versus φ |
| `make_results.py` | Main table and figures from the result CSVs |
| `secagg_check.py` | Numerical check of the secure aggregation protocol (pure NumPy) |
| `requirements.txt` | Python dependencies (PyTorch is installed separately) |

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/<your-account>/FedOSSA.git
cd FedOSSA
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

# 1) PyTorch for your GPU. RTX 50-series (Blackwell) needs the CUDA 12.8 build:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
#    other GPUs / CPU: see https://pytorch.org/get-started/locally/

# 2) everything else
python -m pip install -r requirements.txt
```

Optional: set `HF_HOME` to choose where models and datasets are cached, and `HF_TOKEN`
if you use gated models.

## Quick start

Every script runs with **no arguments** using the default configuration below. Run the
three studies, then build the table and figures:

```bash
python run_sweep.py      # main comparison      -> results/part_<model>/sweep_results.csv + sweep_curves.json
python run_dp.py         # aggregation noise    -> results/part_<model>/dp_sweep.csv
python run_async.py      # stragglers           -> results/part_<model>/async.csv
python make_results.py   # table + figures      -> results/summary/
python secagg_check.py   # secure aggregation   -> results/secagg_check.csv
```

The launchers train the LLMs one after another, each in its own process so that GPU
memory is released between models.

**Resuming.** Every finished run is written immediately as one CSV row. If a run is
interrupted (Ctrl+C, crash, shutdown), run the same command again: finished runs are
skipped and only the interrupted run starts over. Do not delete the result CSVs, since
they are the record of what has finished.

## Default configuration

| Setting | Value |
|---|---|
| Clients K | 8, equal weights pₖ = 1/K |
| Data partition | Dirichlet, concentration 0.1 (strongly heterogeneous) |
| Samples per client | 128 (validation 128, test 256, all disjoint) |
| Rounds | 30, one local pass per round, batch size 4 |
| Optimizer | AdamW, learning rate 2e-4 (LoRA) |
| LoRA | rank 16, scaling 32; fused attention projection (GPT-2, BLOOM), query and value projections (Qwen3) |
| Maximum sequence length | 256 |
| Seeds | 0, 1, 2 |
| Aggregation noise ρ (main comparison, stragglers) | 0.1 |
| Noise grid (`run_dp.py`) | ρ ∈ {0, 0.2, 0.4, 0.6, 0.8} |
| Straggler grid (`run_async.py`) | φ ∈ {0.25, 0.5, 0.75}, delay 2 rounds |
| FedPTuning | 20 virtual tokens, learning rate chosen by validation from {3e-3, 1e-2, 3e-2} |
| Evaluation | best validation round; test perplexity; CIDEr-D on 100 generated responses |

Any setting can be changed from the command line, for example:

```bash
python run_sweep.py --models gpt2-l --methods fedossa fedossa_secagg fedavg
python run_dp.py --seeds 0 --dp_grid 0 0.4 0.8
python run_async.py --models qwen3-1.7b bloom-1b1 --methods fedossa fedask
python run_sweep.py --child --help          # list every option
```

## Outputs

All results go to `results/part_<model>/`.

| File | One row per | Main columns |
|---|---|---|
| `sweep_results.csv` | seed × dataset × method | `perplexity`, `cider`, `train_loss_final`, `best_round`, `uplink_mb`, `trainable_params`, `lr` |
| `sweep_curves.json` | seed × dataset × method | training and validation loss for every round |
| `dp_sweep.csv` | seed × dataset × method × ρ | `perplexity`, `perplexity_final`, `mean_recoverable_rank` |
| `async.csv` | seed × dataset × method × φ × mode | `perplexity`, `eval_loss`; mode = `sync`, `async_naive`, `async_weighted` |
| `fedptuning_lr_check.csv`, `fedptuning_lr_<model>.json` | learning rate tried | validation loss of each rate and the chosen rate |

`make_results.py` writes to `results/summary/`:

| File | Content |
|---|---|
| `table_main.csv` | test perplexity, CIDEr-D and final round training loss per LLM and method |
| `fig_training_loss.pdf/.eps`, `training_loss.csv` | training loss per communication round |
| `fig_robustness.pdf/.eps`, `noise.csv`, `stragglers.csv` | perplexity versus ρ (a) and versus φ (b); solid = stale updates equally weighted, dashed = staleness weighted |

Values are means over the seeds and the three datasets. A cell is reported only when
all of its runs exist; nothing is interpolated. Use `python make_results.py --seeds 0` for a
single seed.

## Method details

**FedOSSA.** For each adapted module the server forms G = Σₖ pₖ(BₖAₖ + Eₖ). With
m = min(d₁, d₂), n = max(d₁, d₂), β = m/n and normalized singular values
yᵢ = σᵢ / (ν√n), each singular value is replaced by ν√n · η(yᵢ), where
η(y) = √((y² − β − 1)² − 4β) / y for y > 1 + √β and 0 otherwise. The new global factors
are B = U_r diag(σ̂₁…σ̂_r) and A = V_rᵀ. When ν = 0 this reduces to the best rank r
approximation of the exact aggregate.

**Aggregation noise.** Every client upload receives Gaussian noise of standard deviation
ρ times the mean standard deviation of the client updates, so ρ is comparable across
models. For FedAvg / FedIT the noise is added to the uploaded factors; for the product
space methods it is added to the product updates; for FedPTuning it is added to the
uploaded prompts. ρ is a robustness knob, not a formal (ε, δ) differential privacy
guarantee.

**Stragglers.** A fraction φ of the clients is delayed by two rounds and re-sends its
cached update. `async_naive` weights stale updates like fresh ones; `async_weighted`
uses pₖ ∝ 1/(1 + delay). The straggling clients are drawn from a seed that depends only
on (seed, dataset, φ), so **every method sees the same straggling clients**.

**FedOSSA+SecAgg.** Pairwise masks cancel exactly in the sum, so the server recovers the
same aggregate as FedOSSA and the training results of the two variants are identical.
In the training code, secure aggregation is represented at the level of this aggregate;
the cost that differs is communication: masking requires the dense product BₖAₖ to be
uploaded, so `uplink_mb` for `fedossa_secagg` is the dense product size (about 55× to 72×
the LoRA factors for the three LLMs at rank 16). `secagg_check.py` runs the full protocol
numerically (pairwise seeds, PRG masks over ℤ_q, fixed point encoding, centred decoding)
and shows that the masks cancel, that the decoding error stays within K · 2^(−f−1), and
that FedOSSA on the decoded sum converges to FedOSSA on the exact sum as the precision f
grows (relative difference about 1e‑4 at f = 24 and 5e‑7 at f = 32).

**FedPTuning.** The soft prompt is initialised from the input embeddings of randomly
sampled vocabulary tokens, so it starts on the embedding manifold. Its learning rate is
chosen once per LLM by the lowest validation loss on Dolly (first seed, no aggregation
noise); the test set is never used for this choice. Use `--prompt_lr` to fix it.

**FedIT.** FedIT applies FedAvg to the LoRA factors, so under this common protocol the two
coincide; by default FedAvg is run once and reported as "FedAvg / FedIT".

## Reproducibility notes

* All randomness (data shuffles, Dirichlet partitions, initialisations, noise, straggler
  sets) is seeded. Runs on a GPU can still differ in the last digits because of
  non deterministic CUDA kernels.
* Test metrics are always taken at the best validation round; the final round values are
  also stored (`perplexity_final`).
* On Windows, if a result CSV is open in Excel when a run finishes, the row is saved to a
  `*_recovered_*.csv` file next to it and is picked up automatically.
* Tested on a workstation with an Intel Core Ultra 9 275HX, 64 GB RAM and a 24 GB NVIDIA
  GPU (CUDA 12.8 build of PyTorch).

## Citation

The paper is under review. Until it is published, please cite:

```bibtex
@misc{mahabub2026fedossa,
  author = {Mahabub, Atik and Vakili, Shervin},
  title  = {{FedOSSA}: Noise Calibrated Spectral Aggregation with Secure Aggregation
            for Federated Fine Tuning of Large Language Models},
  note   = {Under review},
  year   = {2026}
}
```

## Acknowledgment

Supported by The Natural Sciences and Engineering Research Council of Canada (NSERC)
grant RGPIN-2024-06358.

## License

Add a license of your choice (for example MIT or Apache 2.0) as `LICENSE` before making
the repository public. The datasets and model weights are subject to their own licenses on
the Hugging Face Hub.

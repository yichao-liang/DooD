# DooD — Drawing out of Distribution

Reference implementation of **"Drawing out of Distribution with Neuro-Symbolic
Generative Models"** — Yichao Liang, Joshua B. Tenenbaum, Tuan Anh Le,
N. Siddharth. *NeurIPS 2022.*

> Learning general-purpose representations from perceptual inputs is a hallmark
> of human intelligence. We present **D**rawing **o**ut **o**f **D**istribution
> (DooD), a neuro-symbolic generative model of stroke-based drawing that
> operates directly on images, requires no supervision or expensive test-time
> inference, and performs unsupervised amortised inference with a symbolic
> stroke model that enables both interpretability and generalisation. DooD
> transfers zero-shot across MNIST, EMNIST, KMNIST, Quickdraw, and Omniglot,
> outperforming neural baselines, and is competitive with state-of-the-art
> neuro-symbolic models on Omniglot one-shot classification and exemplar
> generation — without requiring stroke-level supervision or data
> augmentation.

The model factorises each image into a sequence of strokes via four modules —
*layout*, *stroke*, *rendering*, *compositing* — trained jointly with a
matching recognition network under an ELBO objective. See the paper for
details.

## Repository layout

```
src/dood/
    __init__.py
    util.py                  shared data loading, checkpointing, utilities
    plot.py                  plotting and tensorboard helpers
    losses.py                ELBO / REINFORCE loss functions
    train.py                 training loop
    evaluate.py              marginal-likelihood and classification evaluation
    classify.py              one-shot classification pipeline
    mws.py                   Memoised Wake-Sleep baseline
    run.py                   training entry point (CLI args)
    sweep.py                 driver for the named experiments in exp_dict.py
    exp_dict.py              registry of named experiment configurations
    run_air_test.py          AIR baseline smoke runs
    run_bl_param_search.py   baseline hyperparameter search
    models/                  generative / recognition networks
        air.py, air_mlp.py, base.py, ssp.py, ssp_mlp.py,
        template.py, vae.py
    datasets/                dataset adapters
        cluster.py, multimnist.py, oneshot_omniglot.py, synthetic.py
    splines/                 differentiable Bezier rendering
        bezier.py, model.py, robust_splines_sklearn.py

scripts/slurm/               SLURM wrappers used for the paper experiments
    execute.sh, beval.sh, bexp.sh, btrain.sh

results/                     artifacts from the paper
    figures/                 cross-dataset reconstructions, tSNE, swarm plots
    metrics/                 eval_clf.csv, eval_mll.csv

tests/                       pytest smoke tests
```

## Installation

This repo requires Python 3.10+ and a recent PyTorch / CUDA toolchain.

### Conda (GPU, matches the paper environment)

```bash
conda env create -f environment.yml
conda activate glot
pip install -e ".[develop]"
```

### uv / pip (Linux CPU or CUDA via PyTorch wheel index)

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e ".[develop]"
```

Install without the dev tooling:

```bash
pip install -e .
```

## Running the code

All scripts live under the `dood` package and should be invoked with
`python -m dood.<module>`.

### Train the full model on MNIST

```bash
python -m dood.sweep -m M --beta 4 -ds mn --seed 0 -trn
```

Flags:

- `-m <code>` — model code from `exp_dict.py` (e.g. `M` = full DooD,
  `AIR` = AIR baseline, `DAIR_l` = Difference AIR).
- `-ds <code>` — dataset (`mn`, `em`, `km`, `qd`, `om`, `sy`).
- `--beta <float>` — β term for the KL over stopping, as in β-VAE.
- `--seed <int>` — random seed.
- `-trn` — run training (omit for evaluation on a trained checkpoint).
- `-ct` — continue training from the latest checkpoint.

### Evaluate a trained checkpoint

```bash
python -m dood.sweep -m M --beta 4 -ds om --seed 0 -it 500000
```

Evaluation writes marginal-likelihood scores to
`results/metrics/eval_mll.csv`, classification scores to
`results/metrics/eval_clf.csv`, and figures under `results/figures/`.

### SLURM wrappers

The wrappers that drove the paper sweeps live in `scripts/slurm/`:

- `execute.sh` — generic `sbatch` template (set `DOOD_ROOT` to this repo).
- `btrain.sh`, `beval.sh`, `bexp.sh` — canned training / evaluation /
  ablation sweeps. They reference `python -m dood.sweep ...` commands.

Edit the `cd` target in `execute.sh` or export `DOOD_ROOT=/path/to/DooD`
before submitting.

## Reproducing the paper

The `results/` directory contains the figures and CSV metrics from the
submission. To rerun a particular experiment, find its code in
`src/dood/exp_dict.py` or `src/dood/sweep.py` and submit the corresponding
line from `scripts/slurm/btrain.sh` (training) followed by
`scripts/slurm/beval.sh` (evaluation).

## Development

```bash
./run_autoformat.sh      # black + docformatter + isort
./run_ci_checks.sh       # mypy + pylint on tests/ + pytest
```

Tooling notes:

- `black`, `isort`, `docformatter` format the whole tree.
- `mypy` runs in a *lenient* mode for `dood.*` (`ignore_errors = True` in
  `pyproject.toml`) and strict mode elsewhere. Tighten incrementally by
  removing the override once modules gain type annotations.
- Pylint (via `pytest-pylint`) is scoped to `tests/` only — the research
  code in `src/dood/` is not linted in CI yet. Extend the linting job in
  `.github/workflows/ci.yml` to cover more modules as they are cleaned.

## Citation

```bibtex
@inproceedings{liang2022dood,
  title     = {Drawing out of Distribution with Neuro-Symbolic Generative Models},
  author    = {Liang, Yichao and Tenenbaum, Joshua B. and Le, Tuan Anh and Siddharth, N.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

## License

MIT — see [LICENSE](LICENSE).

# PAVE: Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic Methods

Official implementation of **PAVE (Policy-Aware Value-field Equalization)**, a critic-centric regularization framework that stabilizes the action-gradient field induced by the Q-function. PAVE is built on top of TD3 and SAC and improves policy smoothness without modifying the actor.

> **Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic Methods.**
> Jeong Woon Lee\*, Kyoleen Kwak\*, Daeho Kim\*, Hyoseok Hwang.
> *International Conference on Machine Learning (ICML), 2026 — Spotlight.*

---

## Overview

Continuous actor–critic policies often produce erratic, high-frequency actions that prevent physical deployment. PAVE shows that this non-smoothness is governed by the **differential geometry of the critic** rather than the actor, and proves a Lipschitz bound

```
L  ≤  M / μ
```

where `M = ‖∇²_sa Q‖` is the mixed-partial volatility (noise sensitivity) and `μ = |λ_max(∇²_aa Q)|` is the action-space curvature (signal distinctness). PAVE stabilizes the Q-gradient field with three lightweight, Hessian-free auxiliary losses on the critic:

| Loss | Role | Estimator |
|------|------|-----------|
| `L_MPR` (Mixed-Partial Regularization) | Suppress `M` isotropically | Finite-difference Taylor proxy with random `ε` |
| `L_VFC` (Vector Field Consistency)     | Suppress `M` along dynamics directions | Finite-difference along `(s_t, s_{t+1})` |
| `L_Curv` (Curvature Preservation)      | Lower-bound `μ`, prevent over-flattening | Hutchinson trace estimator on `∇²_aa Q` |

Total objective:

```
L_total = L_TD  +  λ₁ L_MPR  +  λ₂ L_VFC  +  λ₃ L_Curv
```

The actor remains an unmodified standard TD3 / SAC actor — PAVE only paves the learning signal.

---

## Repository Layout

```
PAVE/
├── td3/                       Twin Delayed DDPG variants
│   ├── models/                base_td3, caps_td3, grad_td3, asap_td3, pave_td3, *_lips_td3
│   └── tests/                 train / eval scripts, helpers, hyperparameter configs
│       ├── modules/           controller, action_extractor, q_extractor, q_extractor2, params, envs
│       ├── test_all.py        TD3 training entry point
│       ├── eval_concavity.py  exact-trace concavity satisfaction
│       ├── eval_q_supinf.py   trajectory-wise sup ‖∇²_sa Q‖₂ and inf |tr ∇²_aa Q|
│       ├── eval_cosine_sim.py cosine similarity of ∇_a Q across timesteps
│       └── eval_wallclock.py  wall-clock convergence from TB logs
├── sac/                       Soft Actor-Critic variants
│   ├── models/                vanilla_sac, caps_sac, grad_sac, asap_sac, pave_sac, *_lips_sac
│   └── tests/                 train / eval scripts (SAC counterparts)
└── README.md
```

Heavy artefacts (pretrained weights, TensorBoard logs, evaluation CSVs, figures) are intentionally **not** included in this repository. Recipes to reproduce them are below.

---

## Setup

### Requirements
- Python 3.10
- PyTorch ≥ 2.1
- Stable-Baselines3 == 2.5.0
- Gymnasium with MuJoCo (`gymnasium[mujoco]`)
- Standard scientific stack (NumPy, SciPy, pandas)

### Install
```bash
conda create -n pave python=3.10
conda activate pave
pip install torch torchvision
pip install "stable-baselines3==2.5.0" "gymnasium[mujoco]"
pip install numpy scipy pandas matplotlib tqdm tensorboard
```

> **Activation note.** PAVE uses **SiLU** activations in the critic to ensure C²-continuity, which is required by the Hutchinson trace estimator in `L_Curv`. All baselines in the paper are also re-trained with SiLU for fair comparison ("SiLU-unified" setting).

---

## Training

### TD3
```bash
cd td3/tests
# Train PAVE on a single environment / seed
python test_all.py --env ant     --algo pave_td3 --seed 178132

# Train all baselines
python test_all.py --env walker  --algo base_td3 --seed 178132
python test_all.py --env walker  --algo caps_td3 --seed 178132
python test_all.py --env walker  --algo grad_td3 --seed 178132
python test_all.py --env walker  --algo asap_td3 --seed 178132
```

### SAC
```bash
cd sac/tests
python test_all.py --env ant --algo pave_sac --seed 178132
```

### Hyperparameters

PAVE weights `(λ₁, λ₂, λ₃)` follow a coordinate-search policy starting from
`(0.1, 0.1, 0.01)`, swept in the order `λ₃ → λ₂ → λ₁`. Per-environment values
are hard-coded in `td3/tests/modules/params.py` and `sac/tests/modules/params.py`,
and are listed in Tables 7–8 of the paper. The perturbation scale `σ = 0.01`
and curvature floor `δ = 1.0` are fixed across all environments.

The seeds used in the paper are stored in `td3/tests/seeds.txt` and
`sac/tests/seeds.txt`; evaluation seeds are in `validation_seeds.txt`.

---

## Evaluation

All evaluation scripts produce CSVs (`re`, `sm`, `M_sup`, etc.) and operate on
trained checkpoints. Replace `<pths_root>` with your local results directory.

```bash
cd td3/tests

# Cumulative return + smoothness score (Tables 1, 2)
python evals.py --pths_root <pths_root> --env walker --algo pave_td3

# Concavity satisfaction rate
python eval_concavity.py --pths_root <pths_root> --env walker --algo pave_td3

# sup ‖∇²_sa Q‖₂ and Neg-Def rate
python eval_q_supinf.py --pths_root <pths_root> --env walker --algo pave_td3

# Cosine similarity of ∇_a Q across timesteps
python eval_cosine_sim.py --pths_root <pths_root> --env walker --algo pave_td3

# Wall-clock convergence time
python eval_wallclock.py --tb_root <tb_root>
```

The smoothness score `sm` is the spectral metric of
[Mysore et al. (2021)](https://arxiv.org/abs/2012.06644) /
[Christmann et al. (2024)](https://arxiv.org/abs/2406.18002), implemented in
`tests/modules/extract.py`.

---

## Reproducing the Paper

Each environment is run for 5 seeds (`178132, 410580, 922852, 787576, 660993`).
The full SiLU-unified TD3 / SAC tables in the paper are reproduced by:

```bash
for env in lunar pendulum reacher ant hopper walker; do
  for algo in base_td3 caps_td3 grad_td3 asap_td3 pave_td3; do
    for seed in 178132 410580 922852 787576 660993; do
      python td3/tests/test_all.py --env $env --algo $algo --seed $seed
    done
  done
done
```

Sensitivity, ablation, and LipsNet variants follow analogous loops; see
`td3/tests/test_*_param.py` and `sac/tests/test_*_param.py` for the
parameterised training entry points used to produce the appendix experiments.

---

## Citation

```bibtex
@inproceedings{lee2026pave,
  title     = {Stabilizing the Q-Gradient Field for Policy Smoothness in Actor-Critic Methods},
  author    = {Lee, Jeong Woon and Kwak, Kyoleen and Kim, Daeho and Hwang, Hyoseok},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## License

Released under the terms in [LICENSE](LICENSE).

# MR_RL — Reinforcement Learning for Magnetic Micro-Robot Control

A physics-based simulation environment and RL training framework for controlling magnetic micro-robots (MRs). The robot is steered by an external rotating magnetic field — the agent learns to choose the field frequency and angle at each timestep to navigate to a target position.

Two control approaches are implemented:
- **DDPG** (Deep Deterministic Policy Gradients) — model-free RL via actor-critic with continuous actions
- **Gaussian Process learning** — model identification from circular trajectories, then GP-guided control

## How the Robot Moves

The MR's motion is governed by:

```
ẋ = a₀ · f · cos(α)
ẏ = a₀ · f · sin(α)
```

where `f` is the field frequency (Hz) and `α` is the field angle (rad). `a₀` is a robot-specific mobility constant that must be identified from data. The simulator integrates these equations using SciPy's **RK45** solver at `dt = 30 ms` per timestep (matching the experimental sensing rate).

## Project Structure

```
MR_RL/
├── MR_simulator.py       # Physics engine — RK45 integration of MR dynamics
├── MR_env.py             # OpenAI Gym environment wrapping the simulator
├── MR_viewer.py          # Matplotlib-based visualizer for trajectories
├── MR_data.py            # Experiment data loader
├── MR_experiment.py      # Interface to pass actions to real hardware and read outputs
├── Learning_module.py    # GP model learning (1D, learns a0 + drift correction)
├── Learning_module_2d.py # GP model learning (2D variant)
├── main.py               # GP learning pipeline: generate data → learn → test
├── main_2d.py            # 2D variant of main
├── utils.py              # Shared utilities: run_sim, plotting helpers
├── RL/
│   ├── MR_ddpg.py        # DDPG agent (actor, critic, replay buffer, OU noise)
│   ├── MR_ppo_keras_rl.py# PPO variant using keras-rl
│   ├── CustomTrackerV10.py # Training callback for logging metrics
│   ├── evaluate_learning.py# Post-training evaluation script
│   └── read_data.py      # Loads saved experiment histories
├── _experiments/         # Saved experiment results from DDPG training runs
├── h5f_files/            # Saved Keras model weights (.h5)
├── old/                  # Archived earlier implementations (DQN, Xbox controller, etc.)
└── lib/                  # Reference Gym environments (gridworld, blackjack, cliff walking)
```

## Gym Environment (`MR_env.py`)

`MR_Env` is an OpenAI Gym-compatible environment that wraps the simulator.

| Property | Value |
|---|---|
| Action space | `Box([0, 0], [20, 2π])` — frequency (Hz) and field angle (rad) |
| Observation space | `(x, y, x_target, y_target, distance)` |
| Max timesteps | 50 per episode |
| Goal | Reach within 30 units of target |
| Boundaries | ±510 units in x and y |

The environment supports both simulation mode (using `MR_simulator.py`) and real-hardware mode (using `MR_experiment.py`).

## GP Learning Pipeline (`main.py`)

The Gaussian Process approach identifies the robot's `a₀` and residual disturbances from data, then uses the learned model to compute optimal control inputs online.

**Steps:**

1. **Estimate drift** — hold the robot still for ~3 seconds; fit a GP to the measured drift velocity `(Dx, Dy)`
2. **Collect training data** — drive the robot in 3 circles over 60 seconds at a fixed frequency; record `(α, vx, vy)` pairs
3. **Learn `a₀`** — fit GPs for residual x/y dynamics; solve for `a₀` that minimises prediction error
4. **Test** — given a desired velocity vector, solve:

```
min_α  (a₀·f·cos(α) + GP_x(α) + Dx − v_desired_x)²
      + (a₀·f·sin(α) + GP_y(α) + Dy − v_desired_y)²
```

To run:

```bash
python main.py
```

Key parameters at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `freq` | 4 Hz | Field frequency for training and testing |
| `a0_def` | 1.5 | Nominal `a₀` used to generate simulated data |
| `dt` | 0.030 s | Timestep (30 ms) |
| `noise_var` | 0.5 | Noise added to simulate model mismatch |
| `cycles` | 3 | Number of training circles |

## DDPG Agent (`RL/MR_ddpg.py`)

A standard DDPG implementation for continuous control:

- **Actor** — fully connected network: `state (5) → 400 → 300 → action (2)` with tanh output
- **Critic** — `state (5) + action (2) → 400 → 300 → Q-value (1)`
- **Replay buffer** — circular deque storing `(s, a, r, done, s')`
- **Ornstein-Uhlenbeck noise** — temporally correlated exploration noise (θ=0.15, σ=0.3)
- **Target networks** — soft updates (τ=0.001) for training stability

To train:

```bash
cd RL
python MR_ddpg.py
```

Saved model weights go to `h5f_files/`. Training histories are saved to `_experiments/`.

## Requirements

```bash
pip install numpy scipy gym tensorflow tflearn scikit-learn matplotlib
```

> **Note:** `RL/MR_ddpg.py` and `RL/MR_ppo_keras_rl.py` use TensorFlow 1.x / TFLearn. If running on TF2, use compatibility mode (`tf.compat.v1`).

## References

- Physics simulation approach: [Simulating Dynamical Systems with Python](https://towardsdatascience.com/a-beginners-guide-to-simulating-dynamical-systems-with-python-a29bc27ad9b1)
- Gym environment structure: [OpenAI Gym from Scratch](https://towardsdatascience.com/openai-gym-from-scratch-619e39af121f)

## Related Work

- [`MRs_suhail`](https://github.com/SuhailSama/MRs_suhail) — MATLAB pipeline using RRT* + nonlinear MPC for the same robot
- [`DQN4MRs`](https://github.com/SuhailSama/DQN4MRs) — DQN grid-world prototype with STREL spec-guided RL extension

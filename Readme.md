**Competitive & Cooperative Snake RL**

This repository contains implementations and training scripts for multi-agent reinforcement learning applied to a 2-player snake game. It includes competitive algorithms (Nash and Deep Nash Q-Learning) and cooperative/single-agent algorithms (DQN, DDQN, Q-Learning, SARSA, Monte Carlo).

**Quick Summary**:
- **Project**: Multi-agent Snake (competitive & cooperative) using classic and deep RL.
- **Languages**: Python 3.8+.
- **Key libs**: PyTorch (for deep models), NumPy, Pygame (optional, for rendering).

**Repository Structure**
- **`Competitive/`**: Competitive two-player environment, algorithms, training and evaluation scripts.
	- `Algos/deep_nash_q/`: Deep Nash Q implementation, training and demos, model checkpoints.
	- `Algos/nash_q/`: Tabular Nash Q implementation and demos.
	- `run_all_experiments.ps1`: PowerShell helper to run competitive experiments end-to-end.
- **`Corporative/`**: Cooperative/single-agent algorithms (DQN, DDQN, Q-Learning, SARSA, Monte Carlo), training/eval scripts and saved models.
- **`Game/`** (under each folder): Game environment code for the snake game.
- **`experiments/`**: Saved experiment outputs and checkpoints (timestamps and logs).

**Environment & Requirements**
- **Python**: 3.8 or newer recommended.
- **GPU**: Optional (PyTorch with CUDA speeds up deep training).
- **Suggested packages** (install via pip): `torch`, `numpy`, `pygame`, `tqdm`.

Install minimal dependencies (example):

```powershell
python -m pip install --upgrade pip
python -m pip install torch numpy pygame tqdm
```

If you use `cmd.exe` the commands are the same (run them in `cmd` or PowerShell).

**High-level Usage**

1) Run all Competitive experiments (PowerShell):

```powershell
cd Competitive
.\run_all_experiments.ps1 -Episodes 3000 -TestEpisodes 100 -SaveInterval 100
```
A similar shell script is present for Co-operative too.

```powershell
cd Corporative
.\run_all_experiments.ps1 
```
2) Train / Test Competitive Nash Q (tabular):

```cmd
python Competitive\Algos\nash_q\train.py --train --episodes 5000 --save-interval 100
python Competitive\Algos\nash_q\train.py --test --test-episodes 100
```

3) Train / Test Deep Nash Q:

```cmd
python Competitive\Algos\deep_nash_q\train_deep.py --train --episodes 3000
python Competitive\Algos\deep_nash_q\train_deep.py --test --test-episodes 100
```

4) Run demos / rendering (watch gameplay):

```cmd
cd Competitive\Algos\nash_q
python render.py

cd ..\deep_nash_q
python render_deep.py
```

5) Cooperative / single-agent algorithms (examples):

For demo and eval:
```cmd
python Corporative\Algos\DQLearning\demo_dqn.py 
python Corporative\Algos\DDQLearning\eval_ddqn.py --model_path=model_path
```

**Model files and checkpoints**
- Deep model checkpoints: `Competitive/Algos/deep_nash_q/models/` (file names like `agent1_deep_nash.pth`).
- Tabular models: `Competitive/Algos/nash_q/models/` (pickled Q-tables, e.g., `agent1_nash.pkl`).
- Cooperative models: `Corporative/Algos/*/models/<timestamp>/`
- Experiment logs and saved checkpoints are under each `models/` subfolder for reproducibility.

**What each folder contains**
- `Competitive/Algos/deep_nash_q/`:
	- `deep_nash_q_agent.py` - agent wrapper for deep Nash Q.
	- `deep_nash_qlearning.py` - training loop and learning logic.
	- `train_deep.py` / `train_deep_nash_q.py` - training entry points.
	- `eval_deep_nash_q.py`, `demo_deep_nash_q.py` - evaluation and demo utilities.
- `Competitive/Algos/nash_q/`:
	- `nash_q_agent.py`, `nash_qlearning.py` - tabular Nash Q agent and learning loop.
	- `train.py`, `eval_nash_q.py`, `demo_nash_q.py` - training/eval/demo scripts.
- `Corporative/Algos/`:
	- Subfolders for `DQLearning`, `DDQLearning`, `QLearning`, `SARSA_On_Policy`, `Monte_Carlo_on_policy`.
	- Each contains `train_*`, `eval_*`, `demo_*`, agent implementations and `models/` for saved runs.

**Configuration & Hyperparameters**
- Many training scripts accept command-line args for: `--episodes`, `--test-episodes`, `--save-interval`, learning rates, hidden sizes, batch sizes, exploration epsilon schedules.
- See the top of each training script (e.g., `train_deep.py`) for available flags and defaults.

**Reproducing Example Results**
- The `Competitive/README.md` contains sample logs and example outputs showing training and test summaries for both Deep Nash and Nash Q runs (see included attachment logs).
- To reproduce a short test run (fast):

```powershell
.\Competitive\run_all_experiments.ps1 -Episodes 200 -TestEpisodes 20
```

This runs a short training/testing cycle and demonstrates saved models and evaluation summaries.

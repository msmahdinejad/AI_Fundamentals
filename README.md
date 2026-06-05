# AI Fundamentals Portfolio

This repository consolidates five assignments from the **Fundamentals and Applications of Artificial Intelligence** course into one clean, documented project portfolio. The original classroom repositories were reviewed, reorganized, and reduced to the source code, notebooks, assets, and configuration needed to understand and run each phase.

## Repository Layout

```text
AI_Fundamentals/
├── docs/
│   └── AI_Fundamentals_Report.tex
├── projects/
│   ├── phase-01-search-regression-annealing/
│   ├── phase-02-mdp-reinforcement-learning/
│   ├── phase-03-adversarial-search-game/
│   ├── phase-04-fol-minesweeper/
│   └── phase-05-llm-fol-mastermind/
├── requirements.txt
└── README.md
```

## Project Summary

| Phase | Topic | Main Techniques | Entry Points |
| --- | --- | --- | --- |
| 1 | Search, regression, and local search | BFS, UCS, IDS/DLS, A*, SGD linear regression, hill climbing, simulated annealing | `search_algorithms/*_main.py`, notebooks |
| 2 | MDP and reinforcement learning | Value iteration, policy precomputation, Q-learning, DQN bonus | `mdp/main.py`, `unknown_environment/main.py`, `dqn_bonus/main.py` |
| 3 | Adversarial game search | Minimax, alpha-beta pruning, heuristic evaluation | `main.py` |
| 4 | First-order logic solver | PyDatalog facts/rules, Minesweeper inference | `main.py`, `manual.py` |
| 5 | LLM plus symbolic reasoning | Mastermind constraints, SWI-Prolog, OpenAI-compatible API integration | `manual_game.py`, `llm_solver.py` |

## Setup

Create a Python environment and install the shared dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with:

```bash
source .venv/bin/activate
```

Some projects open a Pygame window. Run those scripts from their own project directory so the relative asset paths resolve correctly.

## Running the Projects

### Phase 1: Search, Regression, and Annealing

```bash
cd projects/phase-01-search-regression-annealing/search_algorithms
python BFS_main.py
python UCS_main.py
python DLS_main.py
python AStar_main.py
```

The regression and DNA center tasks are notebook-based:

```bash
jupyter notebook projects/phase-01-search-regression-annealing/linear_regression/modeling_hint.ipynb
jupyter notebook projects/phase-01-search-regression-annealing/simulated_annealing/DNA_Center.ipynb
```

### Phase 2: MDP and Reinforcement Learning

```bash
cd projects/phase-02-mdp-reinforcement-learning/mdp
python main.py

cd ../unknown_environment
python main.py

cd ../dqn_bonus
python main.py
```

The Q-learning and DQN scripts can be computationally heavier than the other phases because they train policies over many episodes.

### Phase 3: Adversarial Search Game

```bash
cd projects/phase-03-adversarial-search-game
python main.py
```

The game uses packaged enemy agents from `src/bin`; those binaries are intentionally retained because the provided environment imports them directly.

### Phase 4: FOL Minesweeper

```bash
cd projects/phase-04-fol-minesweeper
python main.py
```

For manual interaction:

```bash
python manual.py
```

### Phase 5: LLM + FOL Mastermind

Manual game:

```bash
cd projects/phase-05-llm-fol-mastermind
python manual_game.py
```

LLM-assisted solver:

```bash
set OPENAI_API_KEY=your_api_key_here
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini
python llm_solver.py
```

On Linux/macOS, use `export` instead of `set`. The LLM solver also requires SWI-Prolog to be installed and available to `pyswip`.

## Documentation

The consolidated LaTeX report is available at:

```text
docs/AI_Fundamentals_Report.tex
```

Render it with:

```bash
cd docs
pdflatex AI_Fundamentals_Report.tex
```

Commit the generated `AI_Fundamentals_Report.pdf` if a rendered PDF copy is required for submission.

## Cleanup Notes

The final repository excludes temporary clone metadata, Python caches, generated plots, screenshots, LaTeX auxiliary files, and hardcoded secrets. Runtime artifacts are ignored through `.gitignore`.

# AI Fundamentals Portfolio 

This repository is the consolidated final portfolio for the **Fundamentals and Applications of Artificial Intelligence** course. It combines five separate assignment repositories into one clean, documented, and runnable project collection.

## Team Members

- [Mohammad Saleh Mahdinejad](https://github.com/msmahdinejad)
- [Fatemeh Sayadzadeh](https://github.com/Ftm-Sayadzadeh)

## Main Documentation

The full technical report is the primary documentation for the project:

- [Rendered PDF report](AI_Fundamentals_Report.pdf)

The report includes a project introduction, team information, reviewed source material, detailed phase-by-phase explanations, algorithm formulas, implementation notes, selected figures/screenshots from the original phase documents, cleanup decisions, and execution instructions.

## What This Repository Contains

The original course work was submitted as separate GitHub Classroom repositories. This repository reorganizes them into a single portfolio:

```text
AI_Fundamentals/
├── AI_Fundamentals_Report.pdf
├── README.md
├── requirements.txt
└── projects/
    ├── phase-01-search-regression-annealing/
    ├── phase-02-mdp-reinforcement-learning/
    ├── phase-03-adversarial-search-game/
    ├── phase-04-fol-minesweeper/
    └── phase-05-llm-fol-mastermind/
```

## Phase Overview

| Phase | Directory | Main Topics | Main Deliverables |
| --- | --- | --- | --- |
| 1 | `phase-01-search-regression-annealing` | Graph search, SGD regression, local search | BFS, UCS, IDS/DLS, A*, asteroid regression notebook, DNA center notebook |
| 2 | `phase-02-mdp-reinforcement-learning` | MDP, value iteration, reinforcement learning | Known-environment policy solver, Q-learning agent, DQN bonus |
| 3 | `phase-03-adversarial-search-game` | Adversarial search | PaintBattle environment and minimax/alpha-beta agent |
| 4 | `phase-04-fol-minesweeper` | First-order logic inference | PyDatalog Minesweeper solver |
| 5 | `phase-05-llm-fol-mastermind` | LLM plus symbolic reasoning | Mastermind environment and validated LLM-Prolog solver |

## Phase 1: Search, Regression, and Local Search

### Search Algorithms

Location:

```text
projects/phase-01-search-regression-annealing/search_algorithms/
```

Implemented scripts:

- `BFS_main.py`: breadth-first search with dynamic enemy-cycle state tracking.
- `UCS_main.py`: uniform-cost search using cumulative path cost and a priority queue.
- `DLS_main.py`: depth-limited search and iterative deepening search.
- `AStar_main.py`: A* search with a weighted target/obstacle/enemy-aware heuristic.

The search environment is stored under:

```text
search_algorithms/env/
```

That folder is required. It contains the grid-world engine, maps, icons, scoring rules, and the `play(...)` wrapper used by all search scripts.

### Linear Regression

Location:

```text
projects/phase-01-search-regression-annealing/linear_regression/modeling_hint.ipynb
```

This notebook implements an asteroid diameter prediction workflow:

- dataset loading and preprocessing,
- feature scaling and encoding,
- train/validation/test splitting,
- SGD linear regression implemented from scratch,
- learning-rate decay, momentum, and early stopping experiments,
- evaluation with `R²`, MAE, and MSE,
- prediction and residual plots.

### DNA Center Finding

Location:

```text
projects/phase-01-search-regression-annealing/simulated_annealing/DNA_Center.ipynb
```

This notebook solves the closest DNA string problem using:

- Hamming distance,
- radius/evaluation calculation,
- neighbor generation,
- hill climbing,
- random-restart hill climbing,
- stochastic hill climbing,
- random-walk hill climbing,
- simulated annealing,
- brute-force comparison for small inputs.

## Phase 2: MDP and Reinforcement Learning

Location:

```text
projects/phase-02-mdp-reinforcement-learning/
```

This phase uses a stochastic WALL-E grid environment with buildings, enemies, trash, power thresholds, a plant goal, and movement uncertainty.

### Known Environment: MDP

Directory:

```text
mdp/
```

Main files:

- `environment.py`: stochastic grid-world environment.
- `main.py`: value-iteration solver with policy precomputation.
- `main2.py`: alternate value-iteration implementation.

Main implemented ideas:

- value iteration,
- transition-model use,
- goal, trash, and collection reward maps,
- threshold-specific policies,
- risk-aware policy selection,
- stuck detection,
- heat-map and convergence-plot generation.

### Unknown Environment: Q-Learning

Directory:

```text
unknown_environment/
```

Main implemented ideas:

- tabular Q-learning,
- epsilon-greedy exploration,
- state representation as `(position, power_level, trash_mask)`,
- Q-table lazy initialization,
- learning-rate and epsilon decay,
- convergence tracking,
- learned policy map generation.

### Bonus: DQN

Directory:

```text
dqn_bonus/
```

Main implemented ideas:

- PyTorch Dueling DQN,
- replay buffer,
- target network,
- vectorized state representation,
- epsilon-greedy neural action selection,
- mini-batch TD learning.

## Phase 3: Adversarial Search Game

Location:

```text
projects/phase-03-adversarial-search-game/
```

Main files:

- `main.py`: runs a PaintBattle match.
- `Minimax1.py`: custom minimax agent.
- `src/env.py`: PaintBattle game environment.
- `src/agent.py`: base agent interface.
- `src/manual_policy.py`: manual controller.
- `src/maps/`: map definitions.
- `src/bin/`: packaged enemy agents required by the provided environment.

The custom agent implements:

- depth-limited minimax,
- alpha-beta pruning,
- move ordering,
- state hashing/cache support,
- BFS distance estimation,
- mobility counting,
- heuristic evaluation over score, survival, territory, mobility, and opponent pressure.

The binary enemy-agent files in `src/bin/` are intentionally preserved. The game imports them directly, so removing them would break the original assignment environment.

## Phase 4: FOL Minesweeper

Location:

```text
projects/phase-04-fol-minesweeper/
```

Main files:

- `main.py`: automated PyDatalog solver.
- `manual.py`: manual controller.
- `src/minesweeper.py`: Pygame Minesweeper environment.

The solver represents the board with facts such as:

- `hidden(R, C)`
- `revealed(R, C)`
- `flagged(R, C)`
- `neighbor(R, C, NR, NC)`
- `clue(R, C, V)`

It derives:

- safe cells,
- mine cells,
- zero-clue safe expansions,
- frontier cells for efficient querying.

When deterministic inference cannot find a move, the solver falls back to a probability-guided guess.

## Phase 5: LLM + FOL Mastermind

Location:

```text
projects/phase-05-llm-fol-mastermind/
```

Main files:

- `manual_game.py`: interactive Mastermind game.
- `llm_solver.py`: LLM + SWI-Prolog solver.
- `src/mastermind.py`: game environment.

The LLM solver:

- dynamically writes `logic.pl`,
- defines valid codes and scoring predicates in Prolog,
- asks an OpenAI-compatible model to generate turn constraints,
- validates the generated rule before assertion,
- queries Prolog for remaining candidates,
- asks the LLM to choose among candidates,
- falls back to deterministic/random valid choices when needed.

Configuration is intentionally read from environment variables:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

No API key is stored in source code.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- Pygame projects should be run from their own phase directories because they load maps and images with relative paths.
- Phase 5 LLM solver requires SWI-Prolog installed and available on `PATH`.
- The DQN bonus requires PyTorch and can take longer to run than the other projects.

## Running the Projects

### Phase 1 Search

```bash
cd projects/phase-01-search-regression-annealing/search_algorithms
python BFS_main.py
python UCS_main.py
python DLS_main.py
python AStar_main.py
```

### Phase 1 Notebooks

```bash
jupyter notebook projects/phase-01-search-regression-annealing/linear_regression/modeling_hint.ipynb
jupyter notebook projects/phase-01-search-regression-annealing/simulated_annealing/DNA_Center.ipynb
```

### Phase 2 MDP

```bash
cd projects/phase-02-mdp-reinforcement-learning/mdp
python main.py
```

### Phase 2 Q-Learning

```bash
cd projects/phase-02-mdp-reinforcement-learning/unknown_environment
python main.py
```

### Phase 2 DQN Bonus

```bash
cd projects/phase-02-mdp-reinforcement-learning/dqn_bonus
python main.py
```

### Phase 3 PaintBattle

```bash
cd projects/phase-03-adversarial-search-game
python main.py
```

### Phase 4 Minesweeper Solver

```bash
cd projects/phase-04-fol-minesweeper
python main.py
```

Manual version:

```bash
python manual.py
```

### Phase 5 Mastermind

Manual game:

```bash
cd projects/phase-05-llm-fol-mastermind
python manual_game.py
```

LLM solver on Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4o-mini"
python llm_solver.py
```

Linux/macOS:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
python llm_solver.py
```

## Documentation Rendering

The rendered report is stored at the repository root:

```text
AI_Fundamentals_Report.pdf
```

The local LaTeX source is kept outside Git tracking at:

```text
docs/AI_Fundamentals_Report.tex
```

Render it locally with:

```bash
cd docs
pdflatex AI_Fundamentals_Report.tex
pdflatex AI_Fundamentals_Report.tex
move AI_Fundamentals_Report.pdf ..\AI_Fundamentals_Report.pdf
```

## Cleanup Decisions

The consolidated repository intentionally excludes:

- nested `.git` folders from the original classroom clones,
- generated training outputs,
- generated screenshots,
- Python cache folders,
- LaTeX auxiliary files,
- local LaTeX source and extracted figure files,
- temporary `logic.pl`,
- hardcoded API credentials.

It intentionally preserves:

- notebooks,
- maps,
- icons,
- Pygame environment assets,
- PaintBattle binary enemy agents,
- rendered PDF report at the repository root.

## Verification Performed

The repository was checked with:

```bash
python -m compileall projects
```

The notebooks were checked as valid JSON, the LaTeX report was rendered with `pdflatex`, and the repository was scanned to ensure the previously hardcoded LLM API credential was not present in the final source.

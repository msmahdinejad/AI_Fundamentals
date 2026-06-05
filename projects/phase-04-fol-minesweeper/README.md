# Phase 4: FOL Minesweeper

This phase solves Minesweeper using first-order logic-style facts and rules implemented with PyDatalog.

The solver maintains revealed, hidden, flagged, clue, neighbor, and frontier facts. It derives safe cells and mine cells from local constraints, then falls back to probability-guided guesses when no deterministic logical move is available.

Use `main.py` for the automated solver and `manual.py` for manual gameplay.

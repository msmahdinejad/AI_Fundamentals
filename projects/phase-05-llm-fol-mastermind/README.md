# Phase 5: LLM + FOL Mastermind

This phase combines a Mastermind game environment with symbolic constraints and an OpenAI-compatible LLM interface.

- `manual_game.py` runs an interactive Mastermind game.
- `llm_solver.py` creates a Prolog knowledge base, asks the LLM to translate feedback into constraints, validates generated rules, and queries SWI-Prolog for candidate codes.

Set `OPENAI_API_KEY` before running the LLM solver. `OPENAI_BASE_URL` and `OPENAI_MODEL` are optional and default to the standard OpenAI-compatible endpoint and `gpt-4o-mini`.

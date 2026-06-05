"""
Mastermind Solver - LLM + Logical Reasoning
Fundamentals of AI Course

University of Isfahan - Winter 2025

Course Instructor: Professor Dr. Hossein Karshenas
Teaching Assistants:
    - Mahdi Mahdieh
    - Younes Rad
    - Pooya Esfandany
    - Danial Shafiei

This project implements an intelligent Mastermind solver using large language models (LLM)
 and symbolic reasoning (Prolog).
The system combines natural language understanding with logical constraint satisfaction to
 efficiently deduce the secret code.

Key Components:
- Mastermind: Core game environment for code validation and feedback
- LLM Interface: Generates Prolog constraints from natural language game feedback
- Prolog Engine (SWI-Prolog): Performs symbolic reasoning and candidate elimination
- Solver Loop: Integrates LLM, Prolog, and game logic for iterative guessing

Implementation Features:
- Translates game feedback (Correct/Misplaced/Incorrect) into Prolog rules via LLM
- Uses constraint satisfaction to eliminate invalid code combinations
- Avoids brute-force search by leveraging logical deduction
- Supports customizable alphabets and code lengths

Usage:
    from mastermind import Mastermind
    game = MastermindSolver(alphabet="ABCDE", code_len=5)
    correct, misplaced, incorrect = game.guess('ABC')
"""

import random
from collections import Counter

class Mastermind:
    def __init__(self, alphabet, code_len, max_turn, seed=None):
        self.alphabet = list(alphabet)
        self.code_len = code_len
        self._max_turn = max_turn
        self._seed = seed
        
        self.reset()
        
    def reset(self):
        self._turn = 0
        self._random_state()
        self._generate_secret()

    def _random_state(self):
        seed = self._seed
        if seed is None:
            seed = int(random.random())
        random.seed(seed)

    def _generate_secret(self):
        self._secret = "".join(random.choice(self.alphabet) for _ in range(self.code_len))
    

    def guess(self, guess):
        guess = guess.upper()
        if len(guess) != self.code_len:
            raise ValueError(f"Guess must be length {self.code_len}")
        if any(ch not in self.alphabet for ch in guess):
            invalid_chars = ", ".join(ch for ch in guess if ch not in self.alphabet)
            raise ValueError(f"Wrong alphabet: {invalid_chars}")
        
        self._turn += 1
        if self._turn > self._max_turn:
            raise ValueError(f"Too many turns. Max turn is {self._max_turn}")
        
        correct = sum(s == g for s, g in zip(self._secret, guess))
        secret_rest = [s for s, g in zip(self._secret, guess) if s != g]
        guess_rest = [g for s, g in zip(self._secret, guess) if s != g]
        
        cs = Counter(secret_rest)
        cg = Counter(guess_rest)
        misplaced = sum(min(cs[ch], cg[ch]) for ch in cg.keys())
        
        return correct, misplaced, self.code_len - correct - misplaced

import sys
import random
import re
import time
import json
import os
from typing import List, Set
from pyswip import Prolog
from openai import OpenAI
from src.mastermind import Mastermind

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROLOG_FILE = "logic.pl"

LEVELS = {
    "BASE": {
        "ALPHABET": list("ABC"),
        "CODE_LEN": 3,
        "MAX_TURN": 5
    },
    "STANDARD": {
        "ALPHABET": list("ABCDE"),
        "CODE_LEN": 5,
        "MAX_TURN": 12
    },
    "ADVANCED": {
        "ALPHABET": list("ABCDEFG"),
        "CODE_LEN": 6,
        "MAX_TURN": 20
    }
}

SELECTED_LEVEL = "ADVANCED"

CONFIG = LEVELS[SELECTED_LEVEL]
ALPHABET = CONFIG["ALPHABET"]
CODE_LEN = CONFIG["CODE_LEN"]
MAX_TURN = CONFIG["MAX_TURN"]

def create_logic_file(alphabet: list, length: int, filename=PROLOG_FILE):
    colors_pl = ",".join([f"'{c}'" for c in alphabet])
    prolog_code = f"""
    color(C) :- member(C, [{colors_pl}]).
    code_length({length}).

    valid_code(Code) :-
        code_length(N),
        length(Code, N),
        maplist(color, Code).

    calc_score(Secret, Guess, C, M, I) :-
        calc_correct(Secret, Guess, C),
        remove_correct(Secret, Guess, S_Rest, G_Rest),
        calc_misplaced(S_Rest, G_Rest, M),
        code_length(Len),
        I is Len - C - M.

    calc_correct([], [], 0).
    calc_correct([H|T1], [H|T2], N) :- 
        !, calc_correct(T1, T2, N1), N is N1 + 1.
    calc_correct([_|T1], [_|T2], N) :- 
        calc_correct(T1, T2, N).

    remove_correct([], [], [], []).
    remove_correct([H|T1], [H|T2], SR, GR) :- 
        !, remove_correct(T1, T2, SR, GR).
    remove_correct([H1|T1], [H2|T2], [H1|SR], [H2|GR]) :- 
        remove_correct(T1, T2, SR, GR).

    calc_misplaced(Secret, Guess, M) :-
        findall(C, color(C), Colors),
        count_colors(Colors, Secret, S_Counts),
        count_colors(Colors, Guess, G_Counts),
        sum_min(S_Counts, G_Counts, M).

    count_colors([], _, []).
    count_colors([C|RestColors], List, [Count|RestCounts]) :-
        occurrences(C, List, Count),
        count_colors(RestColors, List, RestCounts).

    occurrences(_, [], 0).
    occurrences(X, [X|T], N) :- !, occurrences(X, T, N1), N is N1 + 1.
    occurrences(X, [_|T], N) :- occurrences(X, T, N).

    sum_min([], [], 0).
    sum_min([C1|T1], [C2|T2], Sum) :-
        sum_min(T1, T2, Rest),
        Min is min(C1, C2),
        Sum is Rest + Min.
    """
    with open(filename, "w") as f:
        f.write(prolog_code)
    print(f"[Init] Prolog logic file updated for {SELECTED_LEVEL} mode.")

class LogicLMAgent:
    def __init__(self, api_key: str, base_url: str):
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running the LLM solver."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_initial_guess(self) -> str:
        system_prompt = (
            "You are a Mastermind expert. "
            f"Config: Length={CODE_LEN}, Alphabet={ALPHABET}. "
            "Your Goal: Provide the best starting guess to maximize entropy.\n"
            "Process: Think step-by-step about patterns (e.g. AABBCC vs ABCDEF), then select."
        )
        user_prompt = "Output JSON: {\"reasoning\": \"...\", \"selection\": \"CODE\"}"

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            guess = data.get("selection", "").upper().strip()
            reasoning = data.get("reasoning", "")
            
            if len(guess) == CODE_LEN and all(c in ALPHABET for c in guess):
                print(f"   [LLM Initial] Start: {guess}")
                print(f"      Strategy: {reasoning[:100]}...")
                return guess
            else:
                print(f"   [LLM Init Warning] Generated invalid guess: {guess}. Using fallback.")
                
        except Exception as e:
            print(f"   [LLM Init Error] {e}")
        
        fallback = "".join(ALPHABET[:min(len(ALPHABET), CODE_LEN)])
        if len(fallback) < CODE_LEN: fallback += ALPHABET[0] * (CODE_LEN - len(fallback))
        print(f"   [Source: Script] Fallback Initial Guess: {fallback}")
        return fallback

    def generate_prolog_rule(self, turn: int, guess: str, correct: int, misplaced: int, incorrect: int) -> str:
        guess_list = "[" + ",".join(f"'{char}'" for char in guess) + "]"
        safe_fallback = f"turn_{turn}(Candidate) :- calc_score(Candidate, {guess_list}, {correct}, {misplaced}, {incorrect})"
        
        system_prompt = (
            "You are a logic translator for Mastermind. "
            "Task: Convert game feedback into a SWI-Prolog rule.\n"
            "Steps:\n"
            "1. Analyze the feedback (Correct, Misplaced, Incorrect).\n"
            "2. Formulate the logical constraint.\n"
            "3. Write the Prolog code using `calc_score/5`."
        )
        
        user_prompt = (
            f"Turn: {turn}\n"
            f"Guess: {guess_list}\n"
            f"Result: {correct} Correct, {misplaced} Misplaced, {incorrect} Incorrect.\n"
            f"Output Format:\n"
            f"Reasoning: <Analysis>\n"
            f"Code: turn_{turn}(Candidate) :- calc_score(Candidate, {guess_list}, {correct}, {misplaced}, {incorrect})."
        )

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            
            match = re.search(r"(turn_\d+\([A-Za-z0-9_]+\)\s*:-\s*calc_score\(.*\))", content)
            if match:
                rule = match.group(1)
                
                score_match = re.search(r'calc_score\([^)]+,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rule)
                
                if score_match:
                    llm_c, llm_m, llm_i = map(int, score_match.groups())
                    
                    if llm_c + llm_m + llm_i != CODE_LEN:
                        print(f"   [Validation Error] Sum {llm_c+llm_m+llm_i} != {CODE_LEN}")
                        return safe_fallback
                        
                    if any(x < 0 for x in [llm_c, llm_m, llm_i]):
                        print(f"   [Validation Error] Negative scores detected")
                        return safe_fallback

                    if (llm_c, llm_m, llm_i) != (correct, misplaced, incorrect):
                        print(f"   [Validation Error] LLM hallucinated scores ({llm_c},{llm_m},{llm_i}) != Actual ({correct},{misplaced},{incorrect})")
                        return safe_fallback
                else:
                    print("   [Validation Warning] Could not parse scores from LLM rule. Using fallback.")
                    return safe_fallback

                if rule.endswith("."): rule = rule[:-1]
                return rule
            
            return safe_fallback

        except Exception as e:
            print(f"[LLM Error] Rule Gen: {e}")
            return safe_fallback

    def select_best_candidate(self, candidates: List[str], turn: int) -> str:
        if len(candidates) <= 1:
            print(f"   [Source: Script] Only 1 candidate left. Auto-selected: {candidates[0]}")
            return candidates[0]

        sample_size = 60 
        sample_candidates = candidates if len(candidates) <= sample_size else random.sample(candidates, sample_size)
        candidate_str = ", ".join(sample_candidates)

        system_prompt = (
            "You are a strategic Mastermind solver. "
            "Goal: Select the single best code from the list to reduce the remaining search space (Entropy).\n"
            "Constraint: You MUST select one of the provided candidates.\n"
            "Response Format: JSON object."
        )

        user_prompt = (
            f"Turn {turn}.\n"
            f"Candidates (Subset of {len(candidates)}):\n[{candidate_str}]\n\n"
            "Steps:\n"
            "1. Think step-by-step: which candidate is most likely to distinguish between others?\n"
            "2. Select that code.\n"
            "Output JSON: {\"reasoning\": \"Chain of thought...\", \"selection\": \"CODE\"}"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.5,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                data = json.loads(content)
                
                reasoning = data.get("reasoning", "No reasoning provided.")
                selection = data.get("selection", "").upper().strip()

                if selection in candidates:
                    print(f"   [Source: LLM] Selected: {selection}")
                    print(f"      CoT: {reasoning[:120]}...")
                    return selection
                else:
                    print(f"   [LLM Warning] Selected '{selection}' not in candidates. Retrying...")

            except Exception as e:
                print(f"   [LLM Error] Attempt {attempt+1}: {e}")
                continue

        fallback = random.choice(candidates)
        print(f"   [Source: Script] LLM failed. True random selection: {fallback}")
        return fallback

class PrologSolver:
    def __init__(self, logic_file: str):
        self.prolog = Prolog()
        try:
            self.prolog.consult(logic_file)
        except Exception as e:
            print(f"Critical Error: Could not load '{logic_file}'.")
            raise e
            
        self.asserted_rules: Set[str] = set()
        self.turn_rules = []

    def add_constraint(self, rule_str: str, turn: int):
        clean_rule = rule_str.strip()
        if clean_rule.endswith("."): clean_rule = clean_rule[:-1]
        
        if clean_rule in self.asserted_rules:
            pass
        else:
            print(f"   [Prolog] Asserting: {clean_rule}")
            self.prolog.assertz(clean_rule)
            self.asserted_rules.add(clean_rule)
        
        self.turn_rules.append(f"turn_{turn}(C)")

    def solve(self) -> List[str]:
        if not self.turn_rules:
            return []
            
        query = "valid_code(C), " + ", ".join(self.turn_rules)
        try:
            solutions = list(self.prolog.query(query))
        except Exception as e:
            print(f"[Prolog Critical Error] {e}")
            raise RuntimeError("Prolog query failed. Logic file or constraints might be corrupted.")
        
        candidates = []
        for sol in solutions:
            code_str = "".join(str(c) for c in sol['C']).upper()
            candidates.append(code_str)
        return candidates

class GameController:
    def __init__(self):
        create_logic_file(ALPHABET, CODE_LEN, PROLOG_FILE)
        
        random_seed = int(time.time() * 1000)
        print(f"[Init] Using Random Seed: {random_seed}")
        
        self.game = Mastermind(
            alphabet=ALPHABET, 
            code_len=CODE_LEN, 
            max_turn=MAX_TURN, 
            seed=random_seed 
        )
        
        self.agent = LogicLMAgent(api_key=API_KEY, base_url=BASE_URL)
        self.solver = PrologSolver(logic_file=PROLOG_FILE)
        
    def run(self):
        print(f"\n=== Logic-LM Mastermind ({SELECTED_LEVEL} Mode) ===")
        print(f"Config: Length={CODE_LEN}, Alphabet={ALPHABET}, MaxTurns={MAX_TURN}")
        
        current_guess = self.agent.generate_initial_guess()
            
        for turn in range(1, MAX_TURN + 1):
            print(f"\n--- Turn {turn} ---")
            print(f"   [Action] Guessing: {current_guess}")
            
            try:
                correct, misplaced, incorrect = self.game.guess(current_guess)
            except ValueError as e:
                print(f"Error: {e}")
                break
                
            print(f"   [Feedback] Correct: {correct}, Misplaced: {misplaced}, Incorrect: {incorrect}")
            
            if correct == CODE_LEN:
                print(f"\n>>> VICTORY! Found secret in {turn} turns: {current_guess} <<<")
                return

            prolog_rule = self.agent.generate_prolog_rule(turn, current_guess, correct, misplaced, incorrect)
            
            self.solver.add_constraint(prolog_rule, turn)
            
            try:
                candidates = self.solver.solve()
            except RuntimeError as e:
                print(f"System Halt: {e}")
                break
            
            if not candidates:
                print("   [Error] No valid candidates found. Contradiction in constraints?")
                break
            
            print(f"   [Prolog] Remaining candidates: {len(candidates)}")
            
            current_guess = self.agent.select_best_candidate(candidates, turn)
            
        print(f"\nGame Over. Secret was: {self.game._secret}")

if __name__ == '__main__':
    try:
        app = GameController()
        app.run()
    except Exception as e:
        print(f"Critical Error: {e}")

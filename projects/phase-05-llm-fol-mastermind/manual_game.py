from src.mastermind import Mastermind

MAX_TURN = 12
ALPHABET = list("ABCDE")
CODE_LEN = 5


def main():
    mastermind = Mastermind(code_len=CODE_LEN, max_turn=MAX_TURN, alphabet=ALPHABET)

    for turn in range(1, MAX_TURN + 1):
        while True:
            guess = input(f"Turn {turn}/{MAX_TURN}. Enter guess: ").upper()
            if len(guess) == CODE_LEN and all(ch in ALPHABET for ch in guess):
                break
            print(f"Invalid. Use {CODE_LEN} letters from {''.join(ALPHABET)}")

        correct, misplaced, incorrect = mastermind.guess(guess)
        print(f"Correct: {correct}, Misplaced: {misplaced}, Incorrect: {incorrect}")

        if correct == CODE_LEN:
            print(f"You won in {turn} turns!")
            return

    print("Game over. No more turns.")


if __name__ == '__main__':
    main()
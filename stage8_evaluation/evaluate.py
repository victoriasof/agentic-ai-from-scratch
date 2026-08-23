from agent import ask, messages  # your Stage 4 script
from golden_dataset import GOLDEN_DATASET

passed = 0
failed = 0

for case in GOLDEN_DATASET:
    messages.clear()  # fresh conversation for each test, so they don't bleed into each other

    print(f"\nTesting: {case['question']}")
    answer = ask(case["question"])  # you may need to adjust ask() to return the answer string

    success = any(expected.lower() in answer.lower() for expected in case["expected_contains"])

    if success:
        print("  PASS")
        passed += 1
    else:
        print(f"  FAIL — expected one of {case['expected_contains']}, got: {answer}")
        failed += 1

print(f"\n{passed} passed, {failed} failed, out of {len(GOLDEN_DATASET)}")
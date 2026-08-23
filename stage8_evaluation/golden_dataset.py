GOLDEN_DATASET = [
    {
        "question": "Who founded Blorbex, and what year?",
        "expected_contains": ["Alina Kowalski", "2019"],
    },
    {
        "question": "What is Blorbex's motto?",
        "expected_contains": ["Eight Arms, One Vision"],
    },
    {
        "question": "What is 12 multiplied by 8?",
        "expected_contains": ["96"],
    },
    {
        "question": "What color is Blorbex's delivery van?",
        "expected_contains": ["don't know", "no information", "not mentioned"],
        # this one checks that the agent admits it doesn't know,
        # rather than inventing an answer — just as important to test
    },
]
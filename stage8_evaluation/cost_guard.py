PRICE_PER_MILLION_INPUT = 3.00
PRICE_PER_MILLION_OUTPUT = 15.00
DAILY_BUDGET_USD = 1.00

total_spent_today = 0.0


def estimate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * PRICE_PER_MILLION_INPUT
    output_cost = (output_tokens / 1_000_000) * PRICE_PER_MILLION_OUTPUT
    return input_cost + output_cost


def check_budget(input_tokens, output_tokens):
    global total_spent_today
    cost = estimate_cost(input_tokens, output_tokens)
    total_spent_today += cost

    if total_spent_today > DAILY_BUDGET_USD:
        raise RuntimeError(
            f"Daily budget of ${DAILY_BUDGET_USD} exceeded "
            f"(spent ${total_spent_today:.4f}). Stopping."
        )

    print(f"  [this call: ${cost:.4f} | today's total: ${total_spent_today:.4f}]")
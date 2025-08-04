def format_answer(answer):
    # Example: ensure JSON serializable
    if isinstance(answer, set):
        return list(answer)
    return answer
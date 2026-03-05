FEW_SHOT_GSM8K = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "39",
    },
]

FEW_SHOT_CSQA = [
    {
        "question": "What do people use to absorb extra ink from a fountain pen?",
        "choices": {
            "A": "blotter",
            "B": "cloth",
            "C": "desk",
            "D": "floor",
            "E": "paper",
        },
        "answer": "A",
    },
    {
        "question": "What home entertainment equipment requires cable?",
        "choices": {
            "A": "radio",
            "B": "telephone",
            "C": "television",
            "D": "turntable",
            "E": "piano",
        },
        "answer": "C",
    },
    {
        "question": "The fox walked from the city into the forest, what was it looking for?",
        "choices": {
            "A": "pretty flowers",
            "B": "hen house",
            "C": "natural habitat",
            "D": "solid ground",
            "E": "woman",
        },
        "answer": "C",
    },
]


def create_choices_str(choices: dict) -> str:
    """
    Convert a dictionary of choices into a formatted string representation.
    Args:
        choices (dict): A dictionary where keys are choice identifiers and
                       values are choice descriptions.
    Returns:
        str: A newline-separated string where each line contains a choice in
             the format "label) text".
    Example:
        >>> choices = {'a': 'Option A', 'b': 'Option B'}
        >>> create_choices_str(choices)
        'a) Option A\\nb) Option B'
    """

    return "\n".join([f"{k}) {v}" for k, v in choices.items()])


def standard_prompt(question: str, choices: dict = None) -> str:
    """
    Generate a standard zero-shot prompt template for answering questions.
    This function formats a question into a standardized prompt string,
    optionally including multiple choice options for CommonsenseQA dataset.
    Args:
        question (str): The question to be included in the prompt.
        choices (dict, optional): A dictionary of answer choices where keys are choice
            identifiers and values are choice text. If provided, choices will be
            formatted and appended to the prompt. Defaults to None.
    Returns:
        str: A formatted prompt string containing the question and optionally the choices,
            ending with "Answer:" to indicate where the response should follow.
    Example:
        >>> standard_prompt("What is 2+2?")
        'Question: What is 2+2?\nAnswer:'
        >>> standard_prompt("What is the capital of France?", {"A": "Paris", "B": "Lyon"})
        'Question: What is the capital of France?\nChoices:\nA: Paris\nB: Lyon\nAnswer:'
    """

    if choices:
        return (
            f"Question: {question}\n"
            f"Choices:\n{create_choices_str(choices)}\n"
            f"Answer:"
        )

    return f"Question: {question}\nAnswer:"


def fewshot_prompt(question: str, choices: dict = None) -> str:
    """
    Generate a few-shot prompt for either CommonsenseQA or GSM8K tasks.
    This function constructs a prompt with few-shot examples that guide the model to answer multiple choice questions or math problems based on the provided inputs.
    Args:
        question (str): The target question to be answered by the model.
        choices (dict, optional): A dictionary of multiple choice options with their labels.If provided, the prompt is formatted for CommonsenseQA. If None, the prompt is formatted for GSM8K math problems. Defaults to None.
    Returns:
        str: A formatted prompt string containing few-shot examples and the target question.
             - For CommonsenseQA (when choices provided): Returns a prompt asking for the letter of the correct answer.
             - For GSM8K (when choices is None): Returns a prompt asking for the final numeric answer.
    """

    if choices:
        # CommonsenseQA few-shot examples
        examples = ""
        for ex in FEW_SHOT_CSQA:
            examples += (
                f"Question: {ex['question']}\n"
                f"Choices:\n{create_choices_str(ex["choices"])}\n"
                f"Answer: {ex['answer']}\n\n"
            )
        return (
            f"Answer each multiple choice question with only the letter of the correct answer.\n\n"
            f"{examples}"
            f"Question: {question}\n"
            f"Choices:\n{create_choices_str(choices)}\n"
            f"Answer:"
        )

    # GSM8K few-shot examples
    examples = ""
    for ex in FEW_SHOT_GSM8K:
        examples += f"Question: {ex['question']}\n" f"Answer: {ex['answer']}\n\n"
    return (
        f"Answer each math question with only the final numeric answer.\n\n"
        f"{examples}"
        f"Question: {question}\n"
        f"Answer:"
    )


def cot_prompt(question: str, choices: dict = None) -> str:
    """
    Generate a Zero-shot Chain-of-Thought prompt template for reasoning tasks.
    This function creates a prompt string that encourages step-by-step reasoning
    by appending "Let's think this step-by-step" to the question. If multiple choice options are provided, they are formatted and included in the prompt.
    Args:
        question (str): The main question or problem to be reasoned about.
        choices (dict, optional): A dictionary of answer choices to include in the prompt.
            Defaults to None. If provided, choices are formatted and added to the prompt.
    Returns:
        str: A formatted prompt string that includes the question, optional choices,
            and a directive to think step-by-step.
    Example:
        >>> cot_prompt("What is 2+2?")
        'Question: What is 2+2?\nLet\'s think this step-by-step'
        >>> cot_prompt("What is the capital of France?", {"A": "London", "B": "Paris"})
        'Question: What is the capital of France?\nChoices:\n...\nLet\'s think this step-by-step'
    """

    if choices:
        return (
            f"Question: {question}\n"
            f"Choices:\n{create_choices_str(choices)}\n"
            f"Let's think this step-by-step"
        )

    return f"Question: {question}\nLet's think this step-by-step"


def self_consistency_prompt(question: str, choices: dict = None) -> str:
    """
    Generate a chain-of-thought prompt for self-consistency reasoning.
    This uses the same prompt as CoT — the difference is in how
    the experiment script calls the model (5 samples, majority vote).
    Args:
        question (str): The question or task for which to generate a reasoning prompt.
        choices (dict, optional): A dictionary of multiple choice options.
            Defaults to None if not provided.
    Returns:
        str: A formatted prompt string that encourages step-by-step reasoning
            for the given question and optional choices.
    """

    return cot_prompt(question, choices)

import os
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {
    "gemma-2b": "google/gemma-2b-it",
    "gemma-7b": "",
}


def load_model(model_name: str) -> tuple:
    """
    Load a pre-trained language model with its tokenizer and determine the available device.
    This function retrieves a model and tokenizer from the Hugging Face model hub based on the
    provided model name, sets them to evaluation mode, and returns them along with the device
    being used for inference.
    Args:
        model_name (str): The name of the model to load. Must be a key in the MODEL_IDS dictionary.
    Returns:
        tuple: A tuple containing:
            - model (AutoModelForCausalLM): The pre-trained causal language model in evaluation mode.
            - tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
            - device (str): The device being used ('cuda' if available, otherwise 'cpu').
    Raises:
        OSError: If the model or tokenizer cannot be downloaded or loaded from Hugging Face.
    Note:
        - Uses float16 precision on CUDA devices for memory efficiency, float32 otherwise.
        - Automatically maps model layers to available devices using device_map="auto".
        - Sets the model to evaluation mode (no gradients computed).
    """

    model_id = MODEL_IDS[model_name]
    print(f"Loading Model: {model_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    model.eval()

    return model, tokenizer, device


def run_inference(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    num_return_sequences: int = 1,
):
    """
    Run inference on the provided language model with the given prompt.
    Args:
        prompt (str): The input prompt text to generate completions for.
        model (AutoModelForCausalLM): The pre-trained causal language model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        device (str): The device to run inference on (e.g., "cpu", "cuda").
        temperature (float, optional): Sampling temperature for generation.
            If > 0.0, enables sampling; if 0.0, uses greedy decoding. Defaults to 0.0.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 512.
        num_return_sequences (int, optional): Number of independent sequences to generate. Defaults to 1.
    Returns:
        list[str]: A list of decoded generated text completions, excluding the input prompt tokens.
    """

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024  # type: ignore
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0

    generate_kwargs: dict = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    decoded = [
        tokenizer.decode(output[input_length:], skip_special_tokens=True)
        for output in outputs
    ]

    return decoded


def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the numerical answer from a GSM8K response text. This function parses a text response, removes commas from numbers, and extracts the last numerical value found. The extracted number is converted to an integer and returned as a string to match the GSM8K ground truth format.
    Args:
        text (str): The response text potentially containing a numerical answer.
    Returns:
        str | None: The last number found in the text converted to an integer string,
                    or None if no numbers are present in the text.
    Example:
        >>> extract_gsm8k_answer("The answer is 1,234")
        '1234'
        >>> extract_gsm8k_answer("The total comes to 42.5 apples")
        '42'
        >>> extract_gsm8k_answer("No numbers here")
        None
    """

    text = text.replace(",", "")  # Handle "1,234" → "1234"
    numbers = re.findall(r"\d+\.?\d*", text)
    if not numbers:
        return None
    # Return as integer string to match GSM8K ground truth format
    return str(int(float(numbers[-1])))


def extract_csqa_answer(text: str) -> str | None:
    """
    Extract a multiple-choice answer (A-E) from text using pattern matching.
    This function searches for an answer in the provided text using two strategies:
    1. First, it looks for explicit answer indicators (e.g., "the answer is", "answer:") followed by a letter A-E, with optional parentheses.
    2. If no explicit indicator is found, it returns the first standalone letter A-E found in the text.
    Args:
        text: The input text to search for an answer.
    Returns:
        A string containing the uppercase letter (A-E) if an answer is found,
        otherwise None.
    Examples:
        >>> extract_csqa_answer("The answer is (C)")
        'C'
        >>> extract_csqa_answer("answer: B")
        'B'
        >>> extract_csqa_answer("Some text with D in it")
        'D'
        >>> extract_csqa_answer("No answer here")
        None
    """

    match = re.search(
        r"(?:the answer is|answer is|answer:)\s*\(?\s*([A-E])\s*\)?",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # Pattern 2: first standalone A-E letter
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else None


def majority_vote(answers: list[str | None]) -> str | None:
    """
    Determine the most frequently occurring answer from a list of answers for self-consistency prompting. This method is used in self-consistency prompting to aggregate multiple reasoning paths by selecting the most common answer. It filters out None values and returns the answer that appears most often in the valid answers. If all answers are None or the list is empty,
    Args:
        answers: A list of string answers or None values.
    Returns:
        The most common answer string, or None if no valid answers exist.
    Examples:
        >>> majority_vote(["yes", "yes", "no"])
        'yes'
        >>> majority_vote([None, None, "maybe"])
        'maybe'
        >>> majority_vote([None, None])
        None
    """

    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return max(set(valid), key=valid.count)


def consistency_rate(answers: list[str | None]) -> float:
    """
    Calculate the consistency rate of answers based on majority voting. Determines the most common answer among the provided answers using majority vote, and returns the proportion of answers that agree with the majority.
    Args:
        answers: A list of answers, where each answer is a string or None.
    Returns:
        A float representing the consistency rate between 0.0 and 1.0.
        Returns 0.0 if there is no clear majority vote (i.e., vote is None).
        Otherwise, returns the ratio of answers matching the majority vote
        to the total number of answers.
    Examples:
        >>> consistency_rate(['A', 'A', 'B'])
        0.6667
        >>> consistency_rate(['A', 'B', 'C'])
        0.0
    """

    vote = majority_vote(answers)
    if vote is None:
        return 0.0
    agreements = sum(1 for a in answers if a == vote)
    return agreements / len(answers)


def save_results(results: list[dict], filepath: str):
    """
    Save results to a CSV file.
    Args:
        results (list[dict]): A list of dictionaries containing the results to be saved.
        filepath (str): The file path where the CSV file will be saved.
                       Parent directories will be created if they don't exist.
    Examples:
        >>> results = [{'name': 'Alice', 'score': 95}, {'name': 'Bob', 'score': 87}]
        >>> save_results(results, 'output/results.csv')
        Results saved to: output/results.csv
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")

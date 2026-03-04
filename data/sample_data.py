import json
import random
from datasets import load_dataset
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

SEED = 42  # For reproducability
SAMPLE_SIZE = 100


def sample_gsm8K(seed: int, n: int) -> list:
    """
    Load and sample n random items from the GSM8K dataset.
    This function loads the GSM8K (Grade School Math 8K) dataset from Hugging Face,
    randomly selects n samples using the provided seed for reproducibility, and
    extracts the final numerical answer from each sample.
    Args:
        seed (int): Random seed for reproducible sampling.
        n (int): Number of samples to randomly select from the dataset.
    Returns:
        list: A list of dictionaries, each containing:
            - "id" (int): The original index of the item in the dataset.
            - "question" (str): The math problem question.
            - "answer" (str): The final numerical answer extracted from the solution.
    Note:
        GSM8K answers are formatted as "solution text #### <number>".
        This function extracts only the numerical part after "####".
    """

    print("Loading GSM8K Dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    random.seed(seed)
    indices = random.sample(range(len(ds)), n)

    samples = []

    for i in indices:
        item = ds[i]
        raw_answer = item["answer"]
        final_answer = raw_answer.split("####")[-1].strip()
        samples.append(
            {
                "id": i,
                "question": item["question"],
                "answer": final_answer,
            }
        )

    return samples


def sample_commonsenseqa(seed: int, n: int) -> list:

    print("Loading CommonsenseQA...")
    ds = load_dataset("tau/commonsense_qa", split="validation")

    random.seed(seed)
    indices = random.sample(range(len(ds)), n)

    samples = []
    for i in indices:
        item = ds[i]
        # Fomatting the choices in the format of the dictionary {"A":"TextA","B":"TextB"...}
        choices = dict(zip(item["choices"]["label"], item["choices"]["text"]))

        samples.append(
            {
                "id": i,
                "question": item["question"],
                "choices": choices,
                "answer": item["answerKey"],  # e.g. "A", "B", "C"...
            }
        )

    return samples


def write_to_json(data_samples: list, filename: str) -> None:
    """
    Write a list of data samples to a JSON file.
    Args:
        data_samples (list): A list of data samples to be serialized and written to file.
        filename (str): The name of the output file (without the .json extension).
    Returns:
        None
    Side Effects:
        Creates or overwrites a JSON file at "data/{filename}.json" and prints a
        confirmation message to the console.
    """

    with open(f"data/{filename}.json", "w") as file:
        json.dump(
            data_samples,
            file,
            indent=2,
        )
    print(f"Saved: {filename}.json")


if __name__ == "__main__":

    gsm8k_samples = sample_gsm8K(SEED, SAMPLE_SIZE)
    write_to_json(gsm8k_samples, "gsm8k")

    commonsenseqa_samples = sample_commonsenseqa(SEED, SAMPLE_SIZE)
    write_to_json(commonsenseqa_samples, "commonsenseqa")

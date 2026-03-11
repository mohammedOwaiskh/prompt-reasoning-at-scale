import json
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils import (
    extract_csqa_answer,
    extract_gsm8k_answer,
    load_model,
    run_inference,
    save_results,
)

from prompts.templates import fewshot_prompt

MODEL_NAME = "gemma-2b"


def run(model_name: str, dataset_name: str, data_path: str, output_path: str):
    model, tokenizer, device = load_model(model_name)

    with open(data_path, "r") as f:
        data = json.load(f)

    is_gsm8k = dataset_name == "gsm8k"
    results = []

    print(
        f"\nRunning Few-shot Standard Prompting (3-shot) | {model_name} | {dataset_name}"
    )
    print("-" * 60)

    for item in tqdm(data, desc="Evaluating"):
        choices = item.get("choices", None)
        prompt = fewshot_prompt(item["question"], choices)

        # Deterministic: temperature=0
        outputs = run_inference(
            prompt, model, tokenizer, device, temperature=0.0, num_return_sequences=1
        )
        raw_output = outputs[0]

        if is_gsm8k:
            predicted = extract_gsm8k_answer(raw_output)
        else:
            predicted = extract_csqa_answer(raw_output)

        ground_truth = str(item["answer"]).strip()
        correct = predicted == ground_truth

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "ground_truth": ground_truth,
                "raw_output": raw_output,
                "predicted": predicted,
                "correct": correct,
                "strategy": "fewshot",
                "model": model_name,
                "dataset": dataset_name,
            }
        )

    accuracy = sum(r["correct"] for r in results) / len(results) * 100
    print(
        f"\nAccuracy: {accuracy:.2f}% ({sum(r['correct'] for r in results)}/{len(results)})"
    )

    save_results(results, output_path)


if __name__ == "__main__":

    for dataset, data_path in [
        ("gsm8k", "data/gsm8k.json"),
        ("csqa", "data/commonsenseqa.json"),
    ]:
        output_path = f"results/{dataset}_{MODEL_NAME}_fewshot.csv"
        run(MODEL_NAME, dataset, data_path, output_path)

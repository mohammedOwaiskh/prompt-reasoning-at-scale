import argparse
import json
from tqdm import tqdm
from experiments.utils import (
    extract_csqa_answer,
    extract_gsm8k_answer,
    load_model,
    run_inference,
    save_results,
)
from prompts.templates import standard_prompt


def run(model_name: str, dataset_name: str, data_path: str, output_path: str):
    model, tokenizer, device = load_model(model_name)

    with open(data_path, "r") as f:
        data = json.load(f)

    is_gsm8k = dataset_name == "gsm8k"
    results = []

    print(f"\nRunning Zero-shot Standard Prompting | {model_name} | {dataset_name}")
    print("-" * 60)

    for item in tqdm(data, desc="Evaluating"):
        choices = item.get("choices", None)
        prompt = standard_prompt(item["question"], choices)

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
                "strategy": "standard",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gemma-2b", "gemma-7b"])
    args = parser.parse_args()

    for dataset, data_path in [
        ("gsm8k", "data/gsm8k_100.json"),
        ("csqa", "data/csqa_100.json"),
    ]:
        output_path = f"results/{dataset}_{args.model}_standard.csv"
        run(args.model, dataset, data_path, output_path)

import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from prompts.templates import self_consistency_prompt
from utils import (
    load_model,
    run_inference,
    extract_gsm8k_answer,
    extract_csqa_answer,
    majority_vote,
    consistency_rate,
    save_results,
)

NUM_SAMPLES = 5
TEMPERATURE = 0.7
MODEL_NAME = "gemma-2b"


def run(model_name: str, dataset_name: str, data_path: str, output_path: str):
    model, tokenizer, device = load_model(model_name)

    with open(data_path, "r") as f:
        data = json.load(f)

    is_gsm8k = dataset_name == "gsm8k"
    results = []

    print(f"\nRunning Self-Consistency CoT | {model_name} | {dataset_name}")
    print(f"Sampling {NUM_SAMPLES}x per question at temperature={TEMPERATURE}")
    print("-" * 60)

    for item in tqdm(data, desc="Evaluating"):
        choices = item.get("choices", None)
        prompt = self_consistency_prompt(item["question"], choices)

        # Sample NUM_SAMPLES reasoning paths in one call
        raw_outputs = run_inference(
            prompt,
            model,
            tokenizer,
            device,
            temperature=TEMPERATURE,
            num_return_sequences=NUM_SAMPLES,
        )

        # Extract answer from each sampled output
        if is_gsm8k:
            sampled_answers = [extract_gsm8k_answer(o) for o in raw_outputs]
        else:
            sampled_answers = [extract_csqa_answer(o) for o in raw_outputs]

        # Aggregate via majority vote
        final_answer = majority_vote(sampled_answers)
        cons_rate = consistency_rate(sampled_answers)

        ground_truth = str(item["answer"]).strip()
        correct = final_answer == ground_truth

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "ground_truth": ground_truth,
                "sampled_answers": str(sampled_answers),
                "raw_outputs": str(raw_outputs),
                "predicted": final_answer,
                "correct": correct,
                "consistency_rate": round(cons_rate, 3),
                "strategy": "self_consistency",
                "model": model_name,
                "dataset": dataset_name,
            }
        )

    accuracy = sum(r["correct"] for r in results) / len(results) * 100
    avg_consistency = sum(r["consistency_rate"] for r in results) / len(results) * 100
    print(f"\nAccuracy:         {accuracy:.2f}%")
    print(f"Avg Consistency:  {avg_consistency:.2f}%")

    save_results(results, output_path)


if __name__ == "__main__":

    for dataset, data_path in [
        ("gsm8k", "data/gsm8k.json"),
        ("csqa", "data/commonsenseqa.json"),
    ]:
        output_path = f"results/{dataset}_{MODEL_NAME}_self_consistency.csv"
        run(MODEL_NAME, dataset, data_path, output_path)

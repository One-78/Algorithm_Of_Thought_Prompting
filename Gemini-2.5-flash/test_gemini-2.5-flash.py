import json
import math
import os
import re
import time
from collections import Counter
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm


load_dotenv()  # Loads .env file
genai.configure(api_key=os.getenv("OPENAI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def safe_to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def save_results(results, output_file):
    temp_file = output_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(temp_file, output_file)


def load_checkpoint(output_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
            return data, len(data.get("results", []))
    return {"results": [], "accuracy": {}}, 0

# ------------------------------
# Extract numeric answer
# ------------------------------
def extract_answer_from_text(text):
    patterns = [
        r"(?:the answer is|answer:|final answer:)\s*[:\-]?\s*\$?\s*([-+]?\d+(?:\.\d+)?)",
        r"=\s*([-+]?\d+(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower(), re.MULTILINE)
        if matches:
            return safe_to_float(matches[-1])

    lines = text.strip().split("\n")
    if lines:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", lines[-1])
        if numbers:
            return safe_to_float(numbers[-1])

    return None

# ---------------------------------------------
# Algorithm of Thought 
# Hybrid (Algorithm → Python Code → run → ans)
# ---------------------------------------------
def hybrid_aot(question):
    algo_prompt = f"""You are an expert mathematician.
Think step-by-step and describe an ALGORITHMIC PLAN to solve the problem.
Use structured pseudocode style (variables + logic), but DO NOT write Python yet.

Question: {question}

Return only your algorithm plan."""

    algo_response = model.generate_content(algo_prompt)
    algorithm = algo_response.text.strip()

    time.sleep(5)

    aot_prompt = f"""Convert this algorithm into Python code.
The program MUST:
- Define variable 'ans' as the final numeric answer
- No print() or input()
- End with: ans = <final result>

Algorithm:
{algorithm}

# Python code:"""

    code_response = model.generate_content(aot_prompt)
    code_text = code_response.text

    match = re.search(r"```python(.*?)```", code_text, re.S)
    if match:
        code_text = match.group(1)

    local_env = {}
    exec(code_text, {}, local_env)
    result = local_env.get("ans", None)

    return safe_to_float(result)

# ------------------------------
# Self-Consistency CoT 
# ------------------------------
def self_consistency_cot(question, num_samples=5):
    samples = []

    for i in range(num_samples):
        cot_prompt = f"""Solve this math problem step by step. Show your reasoning clearly.
End with "The answer is: [number]"

Question: {question}

Generate a DIFFERENT approach if this is not your first attempt."""

        response = model.generate_content(cot_prompt)
        reasoning = response.text.strip()
        result = extract_answer_from_text(reasoning)

        if result is not None:
            samples.append(result)

        time.sleep(5)

    if samples:
        result_counts = Counter(samples)
        majority_result, _ = result_counts.most_common(1)[0]
        return majority_result

    return None


# ------------------------------
# Evaluate dataset
# ------------------------------
def evaluate_dataset(dataset, output_file="results.json"):
    existing_data, start_idx = load_checkpoint(output_file)
    all_results = existing_data["results"]

    hybrid_correct = sum(1 for r in all_results if r.get("hybrid_correct", False))
    cot_correct = sum(1 for r in all_results if r.get("cot_correct", False))

    total = len(all_results)
    checkpoint_data = {
        "results": all_results,
        "accuracy": {
            "total_questions": total,
            "hybrid_correct": hybrid_correct,
            "hybrid_accuracy": hybrid_correct / total if total > 0 else 0,
            "cot_correct": cot_correct,
            "cot_accuracy": cot_correct / total if total > 0 else 0,
        },
    }

    for idx in tqdm(
        range(start_idx, len(dataset)), initial=start_idx, total=len(dataset)
    ):
        question = dataset[idx]["question"]
        expected = float(dataset[idx]["answer"])

        hybrid_answer = hybrid_aot(question)
        hybrid_is_correct = hybrid_answer is not None and math.isclose(
            hybrid_answer, expected, rel_tol=1e-3
        )
        if hybrid_is_correct:
            hybrid_correct += 1

        time.sleep(5)

        cot_answer = self_consistency_cot(question, num_samples=5)
        cot_is_correct = cot_answer is not None and math.isclose(
            cot_answer, expected, rel_tol=1e-3
        )
        if cot_is_correct:
            cot_correct += 1

        result_entry = {
            "question": question,
            "expected_answer": expected,
            "hybrid_answer": hybrid_answer,
            "hybrid_correct": hybrid_is_correct,
            "cot_answer": cot_answer,
            "cot_correct": cot_is_correct,
        }

        all_results.append(result_entry)

        total = len(all_results)
        checkpoint_data = {
            "results": all_results,
            "accuracy": {
                "total_questions": total,
                "hybrid_correct": hybrid_correct,
                "hybrid_accuracy": hybrid_correct / total,
                "cot_correct": cot_correct,
                "cot_accuracy": cot_correct / total,
            },
        }

        save_results(checkpoint_data, output_file)
        time.sleep(5)

    total = len(all_results)
    print("\nFINAL ACCURACY")
    print(f"Total: {total}")
    print(f"Hybrid: {hybrid_correct}/{total} = {hybrid_correct / total:.2%}")
    print(f"CoT: {cot_correct}/{total} = {cot_correct / total:.2%}")

    return checkpoint_data

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    with open("gsm8K.json", "r") as f:
        test_data = json.load(f)

    # dataset = test_data

    # dataset = test_data[:10]

    dataset = test_data[:250]

    # dataset = test_data[:650]

    # dataset = test_data[:1000]

    results = evaluate_dataset(dataset, output_file="results.json")

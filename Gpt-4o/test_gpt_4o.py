import json
import math
import os
import re
import time
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()  # Loads .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o"   


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


# ------------------------------
# Safe OpenAI generation wrapper
# ------------------------------
def openai_generate(prompt, temperature=0.2, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful reasoning assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=512
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"[Attempt {attempt+1}] Error:", str(e))
            if attempt == max_retries - 1:
                return ""
            sleep_time = 2 * (attempt + 1)
            print(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)


# ---------------------------------------------
# Algorithm of Thought 
# Hybrid (Algorithm → Python Code → run → ans)
# ---------------------------------------------
def hybrid_aot(question):

    algo_prompt = f"""
You are an expert mathematician.
Think step-by-step and write an ALGORITHMIC PLAN to solve the math problem.
Use structured pseudocode but no Python yet.

Question: {question}

Return only the algorithm plan.
"""
    algorithm = openai_generate(algo_prompt, temperature=0.3).strip()

    time.sleep(2)

    aot_prompt = f"""
Convert this algorithm into Python code.

Rules:
- Compute the final numeric answer in variable `ans`.
- No print(), no input().
- End code with: ans = <final_value>

Algorithm:
{algorithm}

Provide only Python code.
"""

    code_text = openai_generate(aot_prompt, temperature=0.2)

    match = re.search(r"```python(.*?)```", code_text, re.S)
    if match:
        code_text = match.group(1)

    local_env = {}
    try:
        exec(code_text, {}, local_env)
        result = local_env.get("ans", None)
        return safe_to_float(result)
    except Exception:
        return None


# ------------------------------
# Self-Consistency CoT
# ------------------------------
def self_consistency_cot(question, num_samples=5):

    samples = []

    for i in range(num_samples):

        cot_prompt = f"""
Solve this math problem. Think step-by-step internally.
Provide a short explanation and end with:
"The answer is: <number>"

If this is not the first attempt, use a different reasoning path.

Question: {question}
"""

        reasoning = openai_generate(cot_prompt, temperature=0.7)
        result = extract_answer_from_text(reasoning)

        if result is not None:
            samples.append(result)

        time.sleep(2)

    if samples:
        freq = Counter(samples)
        return freq.most_common(1)[0][0]

    return None


# ------------------------------
# Evaluate dataset
# ------------------------------
def evaluate_dataset(dataset, output_file="result_gpt_4o.json"):

    existing_data, start_idx = load_checkpoint(output_file)
    all_results = existing_data["results"]

    hybrid_correct = sum(1 for r in all_results if r.get("hybrid_correct", False))
    cot_correct = sum(1 for r in all_results if r.get("cot_correct", False))

    for idx in tqdm(range(start_idx, len(dataset)),
                    initial=start_idx, total=len(dataset)):

        q = dataset[idx]["question"]
        expected = float(dataset[idx]["answer"])

        # HYBRID
        hybrid_ans = hybrid_aot(q)
        hybrid_ok = hybrid_ans is not None and math.isclose(hybrid_ans, expected, rel_tol=1e-3)
        if hybrid_ok:
            hybrid_correct += 1

        time.sleep(2)

        # COT
        cot_ans = self_consistency_cot(q, num_samples=5)
        cot_ok = cot_ans is not None and math.isclose(cot_ans, expected, rel_tol=1e-3)
        if cot_ok:
            cot_correct += 1

        entry = {
            "question": q,
            "expected_answer": expected,
            "hybrid_answer": hybrid_ans,
            "hybrid_correct": hybrid_ok,
            "cot_answer": cot_ans,
            "cot_correct": cot_ok,
        }

        all_results.append(entry)

        save_results({
            "results": all_results,
            "accuracy": {
                "total_questions": len(all_results),
                "hybrid_correct": hybrid_correct,
                "hybrid_accuracy": hybrid_correct / len(all_results),
                "cot_correct": cot_correct,
                "cot_accuracy": cot_correct / len(all_results),
            }
        }, output_file)

        time.sleep(2)

    print("\nFINAL ACCURACY")
    print("Hybrid:", hybrid_correct, "/", len(all_results))
    print("CoT:", cot_correct, "/", len(all_results))

    return all_results


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    with open("gsm8K.json", "r") as f:
        test_data = json.load(f)

    dataset = test_data[:250]   # ADJUST RANGE

    evaluate_dataset(dataset, output_file="result_gpt_4o.json")

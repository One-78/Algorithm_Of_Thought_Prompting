import json
import math
import os
import re
import time
from collections import Counter

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()
# Initialize Groq client
client = Groq(api_key=os.environ.get("groq_api_key"))


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


def save_prompts(prompt_data, output_file="prompts_log.json"):
    """Save algorithm and Python prompts to a JSON file"""
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_prompts = json.load(f)
    else:
        existing_prompts = []

    existing_prompts.append(prompt_data)

    with open(output_file, "w") as f:
        json.dump(existing_prompts, f, indent=2)


def call_groq_model(prompt, temperature):
    """Call Groq model with streaming and collect response"""
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=8192,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Collect streamed response
    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content

    return full_response


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


def hybrid_atp_pot(question, question_idx):
    # Generate algorithm prompt
    algo_prompt = f"""You are an expert mathematician.
Think step-by-step and describe an ALGORITHMIC PLAN to solve the problem.
Use structured pseudocode style (variables + logic), but DO NOT write Python yet.

Question: {question}

Return only your algorithm plan."""

    print(f"Algorithm prompt generated for question {question_idx}")

    # Call Groq model for algorithm
    algorithm = call_groq_model(algo_prompt, temperature=0.3)

    time.sleep(2)  # Rate limiting

    # Generate Python code prompt
    py_prompt = f"""Convert this algorithm into Python code.
The program MUST:
- Define variable 'ans' as the final numeric answer
- No print() or input()
- End with: ans = <final result>

Algorithm:
{algorithm}

# Python code:"""

    print(f"Code prompt generated for question {question_idx}")

    # Call Groq model for code generation
    code_response = call_groq_model(py_prompt, temperature=0.3)

    # Save prompts to JSON
    gpt_oss_120B_prompt_data = {
        "question_idx": question_idx,
        "question": question,
        "algorithm_prompt": algo_prompt,
        "algorithm_response": algorithm,
        "python_prompt": py_prompt,
        "python_response": code_response,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_prompts(gpt_oss_120B_prompt_data)

    # Extract and execute code
    code_text = code_response
    match = re.search(r"```python(.*?)```", code_text, re.S)
    if match:
        code_text = match.group(1)

    try:
        local_env = {}
        exec(code_text, {}, local_env)
        result = local_env.get("ans", None)
        return safe_to_float(result)
    except Exception as e:
        print(f"Error executing code: {e}")
        return None


def self_consistency_cot(question, num_samples=5):
    samples = []

    for i in range(num_samples):
        cot_prompt = f"""Solve this math problem step by step. Show your reasoning clearly.
End with "The answer is: [number]"

Question: {question}

Generate a DIFFERENT approach if this is not your first attempt."""

        response = call_groq_model(cot_prompt, temperature=0.3)
        reasoning = response.strip()
        gpt_oss_120B_cot_prompt_data = {
            "question": question,
            "prompt": cot_prompt,
            "response": reasoning,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_prompts(gpt_oss_120B_cot_prompt_data)
        result = extract_answer_from_text(reasoning)

        if result is not None:
            samples.append(result)

        time.sleep(2)  # Rate limiting

    if samples:
        result_counts = Counter(samples)
        majority_result, _ = result_counts.most_common(1)[0]
        return majority_result

    return None


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

        # Pass the question index to save in prompts
        hybrid_answer = hybrid_atp_pot(question, idx)
        hybrid_is_correct = hybrid_answer is not None and math.isclose(
            hybrid_answer, expected, rel_tol=1e-3
        )
        if hybrid_is_correct:
            hybrid_correct += 1

        time.sleep(2)

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
        time.sleep(2)

    total = len(all_results)
    print("\nFINAL ACCURACY")
    print(f"Total: {total}")
    print(f"Hybrid: {hybrid_correct}/{total} = {hybrid_correct / total:.2%}")
    print(f"CoT: {cot_correct}/{total} = {cot_correct / total:.2%}")

    return checkpoint_data


if __name__ == "__main__":
    with open("gsm8K.json", "r") as f:
        test_data = json.load(f)

    # Select dataset subset
    dataset = test_data

    results = evaluate_dataset(dataset, output_file="gpt_oss_120B_results.json")
    print("\nPrompts have been saved to 'prompts_log.json'")

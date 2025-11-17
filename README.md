# # Algorithm of Thought (AoT) Prompting

## Overview

Algorithm of Thought (AoT) is a two-stage prompting technique that separates **algorithmic planning** from **code implementation**. Instead of asking the model to solve a problem directly, it first creates a structured plan, then converts that plan into executable code.

---

## How It Works

### Stage 1: Algorithmic Planning
The model creates a high-level solution plan using pseudocode or algorithmic thinking, without writing any executable code yet. This stage focuses purely on the logical structure and reasoning steps needed to solve the problem.

### Stage 2: Code Generation & Execution
The model takes the algorithm from Stage 1 and converts it into executable Python code. The code is then run to produce the final numerical answer, stored in a variable called `ans`.

---

## Why Algorithm of Thought Works

### 1. **Logical Thinking First**
   - The model focuses on understanding the problem structure before solving it
   - Creating an algorithm forces clear, step-by-step logical reasoning
   - Separating thinking from calculation prevents rushed or confused solutions
   - Results in more organized and coherent problem-solving approaches

### 2. **Python Does the Math**
   - Python handles all calculations with perfect accuracy
   - No arithmetic mistakes like models make when computing in text
   - Complex operations (loops, formulas, large numbers) are executed reliably
   - The computer does what it's best at: precise computation
  

**Language models calculate by "predicting" what the next number should be - they don't actually compute, they guess based on patterns they've seen.** So when asked "What's 347 × 29?", the model tries to predict what answer looks right, which often leads to mistakes.
**Python actually performs the calculation using precise mathematical operations.** When you run 347 * 29 in Python, it computes the exact result (10,063) using binary arithmetic - there's no guessing involved.
Example:

**Model asked: "What's 123 + 456 + 789 + 234?"**

**Model might output: "1,602" (wrong - it predicted/estimated)**


**Python executes: 123 + 456 + 789 + 234**

**Python outputs: 1602 (correct - it calculated)**



The bigger the numbers or more complex the operation, the worse models get at "predicting" the right answer. Python's accuracy stays perfect regardless of complexity.

---

---

## Performance Results

### GSM8K Dataset Accuracy

| Model | Hybrid AoT Accuracy | Self-Consistency CoT Accuracy |
|-------|---------------------|-------------------------------|
| **Gemini 2.5 Flash** | 96%  | 92% |
| **GPT-3.5 Turbo** | 86%  | 85% |

*Evaluated on 250 problems from the GSM8K mathematical reasoning benchmark*

---


import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def extract_all_code_blocks(text):
    """Extracts all code blocks from AI response wrapped in triple backticks."""
    if "```" in text:
        parts = text.split("```")
        code_blocks = [parts[i].strip() for i in range(1, len(parts), 2)]
        return code_blocks
    return [text.strip()]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if not TOGETHER_API_KEY:
        return jsonify({"error": "API key is not configured. Please add your TOGETHER_API_KEY to the .env file."}), 500

    data = request.json
    prompt = data.get("prompt")
    language = data.get("language")

    if not prompt or not language:
        return jsonify({"error": "A problem description and language are required."}), 400

    full_prompt = f"""
You are an expert {language} programmer.

Given the problem: "{prompt}", write:

1. A complete and runnable **recursive solution**.
2. A complete and runnable **iterative solution**.

Respond with exactly two code blocks, each labeled inside markdown with triple backticks.
Only code should be inside the blocks.
"""

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.5,
        "max_tokens": 2048
    }

    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        raw = response.json()

        content = raw["choices"][0]["message"]["content"]
        code_blocks = extract_all_code_blocks(content)

        recursive = code_blocks[0] if len(code_blocks) > 0 else "// Recursive solution not found"
        iterative = code_blocks[1] if len(code_blocks) > 1 else "// Iterative solution not found"

        return jsonify({
            "recursive_solution": recursive,
            "iterative_solution": iterative
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "The request to the AI service timed out. Please try again."}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {e}"}), 500
    except (KeyError, IndexError) as e:
        return jsonify({"error": f"Failed to parse the AI response. Details: {e}"}), 500

@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    recursive_code = data.get("recursiveCode", "").strip()
    iterative_code = data.get("iterativeCode", "").strip()

    if not recursive_code and not iterative_code:
        return jsonify({"error": "Both code blocks are empty."}), 400

    explain_prompt = f"""
You are a programming tutor. Please explain the two code snippets below to a beginner in a structured and easy-to-understand format.

Format your explanation with these sections:

### 🌀 Recursive Version Explanation
- What it does
- Step-by-step logic
- Example with values
- Pros and cons

### 🔁 Iterative Version Explanation
- What it does
- Step-by-step logic
- Example with values
- Pros and cons

### ⚔️ Recursion vs Iteration
- Key differences
- When to use which

Here are the two code snippets:

#### Recursive Version:
```{recursive_code}```

#### Iterative Version:
```{iterative_code}```
"""


    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": explain_prompt}],
        "temperature": 0.4,
        "max_tokens": 1524
    }

    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return jsonify({"explanation": content})

    except Exception as e:
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)

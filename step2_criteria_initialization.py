import pandas as pd
import ollama

# Load your captions
df = pd.read_csv("D:\lite_ssd\step1_img_caption\captions\captions_cifar10_qwen.csv")
captions = df["caption"].dropna().tolist()

# Use a subset for initialization
sample_text = "\n".join(captions[:20])

prompt = f"""
You are a dataset analyst. Given these image captions:

{sample_text}

Identify common DIMENSIONS (like Action, Background, Object) and list possible ATTRIBUTES under each dimension.
Return output as JSON with structure:
{{
  "Action": ["..."],
  "Background": ["..."],
  "Object": ["..."]
}}
"""

response = ollama.chat(
    model="qwen2.5vl:3b",   # make sure this matches your installed model
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- Criteria Initialization ---")
print(response["message"]["content"])

import json
import ollama  # if you’re calling Qwen via Ollama, otherwise replace with HF pipeline

# Step 2 output (your initialization JSON)
criteria_init = {
  "Action": ["Surprise", "Alarm", "Curiosity", "Mystery"],
  "Background": ["grass", "beach", "trees", "building", "wooden floor", "window", "ocean waves", "red chair", "green plant", "autumn colors"],
  "Object": ["dog", "cat", "ship", "car", "building", "window", "ocean waves", "red chair", "green plant", "autumn colors"]
}

prompt = f"""
You are refining dataset criteria. Here are the initialized dimensions and attributes:

{json.dumps(criteria_init, indent=2)}

Tasks:
- Remove duplicates across dimensions (e.g., 'building' should only appear once in the most appropriate dimension).
- Keep only consistent attributes (drop noisy or irrelevant ones like repeated 'Mystery').
- Return clean JSON only, no explanation, no markdown fences.
"""

outputs = []
for i in range(3):  # run multiple times for consistency
    response = ollama.chat(
        model="qwen2.5vl:3b",  # use Qwen model here
        messages=[{"role": "user", "content": prompt}]
    )
    content = response["message"]["content"].strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("```json")[-1].split("```")[0].strip()
    outputs.append(content)

print("\n--- Raw Refinement Runs ---")
for o in outputs:
    print(o)

refined = {}
for o in outputs:
    try:
        data = json.loads(o)
        for dim, attrs in data.items():
            refined.setdefault(dim, [])
            refined[dim].extend(attrs)
    except Exception as e:
        print("Parse error:", e)

# Deduplicate
for dim in refined:
    refined[dim] = list(set(refined[dim]))

print("\n--- Final Refined Criteria ---")
print(json.dumps(refined, indent=2))

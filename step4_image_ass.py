import pandas as pd
import json

# Load refined criteria
with open("criteria_refined.json", "r") as f:
    criteria = json.load(f)

# Load captions CSV (adjust path if needed)
df = pd.read_csv("D:\\lite_ssd\\step1_img_caption\\captions\\captions_cifar10_qwen.csv")

# Function to assign attributes based on caption text
def assign_attributes(caption, criteria):
    assigned = {"Action": [], "Background": [], "Object": []}
    caption_lower = caption.lower()
    for dim, attrs in criteria.items():
        for attr in attrs:
            if attr.lower() in caption_lower:
                assigned[dim].append(attr)
    return assigned

# Apply to each row
assignments = []
for _, row in df.iterrows():
    caption = str(row["caption"])
    attrs = assign_attributes(caption, criteria)
    assignments.append({
        "filename": row.get("filename", ""),
        "label": row.get("label", ""),
        "caption": caption,
        "Action": ",".join(attrs["Action"]),
        "Background": ",".join(attrs["Background"]),
        "Object": ",".join(attrs["Object"])
    })

# Save results
df_out = pd.DataFrame(assignments)
df_out.to_csv("image_assignments.csv", index=False)

print("✅ Saved assignments to image_assignments.csv")

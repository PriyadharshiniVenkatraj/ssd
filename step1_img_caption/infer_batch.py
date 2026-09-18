import os
import pandas as pd
from datasets import load_dataset
from PIL import Image
import ollama

# Number of images to process
MAX_IMAGES = 100
TEMP_IMAGE_PATH = "temp_cifar.jpg"

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print("Loading CIFAR-10 dataset locally...")
dataset = load_dataset("uoft-cs/cifar10", split="train", streaming=False)

print("\n--- Running Qwen2.5-VL 3B via Ollama ---")

results = []

for i, sample in enumerate(dataset):
    if i >= MAX_IMAGES:
        break

    image = sample['img']            # PIL Image object (32x32)
    label_idx = sample['label']      # Integer index
    true_class = classes[label_idx]  # Friendly label name

    # Upscaling helps the vision encoder interpret macro structures reliably
    resized_image = image.resize((244, 244), Image.Resampling.BICUBIC)
    resized_image.save(TEMP_IMAGE_PATH)

    # Clean, concise formatting instructions optimal for Qwen's vision alignment
    prompt = (
        f"Analyze this image knowing it contains a {true_class}. "
        f"Provide a detailed sentence caption specifically noting the sub-population characteristics "
        f"(such as the object's distinct emotional expression, color variations, and background environment setting)."
    )

    try:
        # Read the raw file bytes to safely stream over local localhost ports
        with open(TEMP_IMAGE_PATH, 'rb') as f:
            image_bytes = f.read()

        # FIXED: Model updated to official Qwen2.5-VL string tag
        response = ollama.chat(
            model='qwen2.5vl:3b',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }],
            options={
                'temperature': 0.1,  # Low temperature for highly precise descriptive indexing
                'num_predict': 60
            }
        )
        caption = response['message']['content'].strip()
    except Exception as e:
        caption = f"Error generating caption: {str(e)}"

    print(f"\nImage #{i+1}:")
    print(f"  Ground Truth Label: {true_class}")
    print(f"  Qwen2.5-VL Caption: \"{caption}\"")

    # Save result for CSV formatting
    results.append({
        "index": i,
        "filename": f"cifar_{i}.jpg",
        "label": true_class,
        "caption": caption
    })

# Clean up temp structural tracks
if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)

# Save all results to CSV
df = pd.DataFrame(results)
os.makedirs("captions", exist_ok=True)
csv_path = os.path.join("captions", "captions_cifar10_qwen.csv")
df.to_csv(csv_path, index=False)

print(f"\nEvaluation complete. Captions saved to {csv_path}")

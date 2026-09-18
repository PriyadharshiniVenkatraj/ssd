import os
import torch
from datasets import load_dataset
from PIL import Image
import ollama

# 1. Load the dataset locally (using your optimized setup)
print("Loading dataset locally...")
dataset = load_dataset("uoft-cs/cifar10", split="train", streaming=False, token=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
TEMP_IMAGE_PATH = "temp_cifar.jpg"

print("\n--- Running Llama 3.2 Vision via Ollama ---")

# 2. Iterate through the first 5 images
for i, sample in enumerate(dataset):
    if i >= 5:
        break
        
    image = sample['img']            # PIL Image object (32x32)
    label_idx = sample['label']      # Integer index
    true_class = classes[label_idx]  # Friendly label name
    
    # 3. Save the in-memory PIL image to a temporary file so Ollama can read it
    image.save(TEMP_IMAGE_PATH)
    
    # 4. Construct a clear visual prompt telling Llama the true category context
    prompt = (
        f"This is a low-resolution image of a {true_class}. "
        f"Provide a single, short descriptive sentence focusing on its primary colors and setting."
    )
    
    try:
        # 5. Send request to your local Ollama instance
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [TEMP_IMAGE_PATH]
            }]
        )
        
        # Extract response text
        caption = response['message']['content'].strip()
        
    except Exception as e:
        caption = f"Error generating caption: {str(e)}"
        
    print(f"\nImage #{i+1}:")
    print(f"  Ground Truth Label: {true_class}")
    print(f"  Llama 3.2 Caption:  \"{caption}\"")

# 6. Clean up the temporary file after finishing the loop
if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)
    
print("\nEvaluation complete.")

LLM‑Driven Dataset Analyst for CIFAR‑10

Overview

This project explores how Large Language Models (LLMs) can act as dataset analysts rather than just caption generators. Using Qwen multimodal models, the pipeline automatically generates captions for CIFAR‑10 images, discovers latent dimensions (Action, Background, Object), refines them, and assigns each image to structured subpopulations.

The goal is to uncover hidden dataset patterns that humans often miss, enabling deeper insights into bias, diversity, and fairness in vision datasets.

Workflow

The pipeline is organized into clear steps:

Image Captioning (step1_img_caption/demo.py, captions_cifar10_qwen.csv)

Generate natural language captions for CIFAR‑10 images using Qwen.

Criteria Initialization (step2_criteria_initialization.py)

Extract candidate dimensions (Action, Background, Object) and attributes from captions.

Criteria Refinement (step3_criteria_refinement.py, criteria_refined.json)

Clean duplicates, stabilize dimensions, and produce a refined JSON schema.

Image Assignment (step4_image_ass.py, image_assignments.csv)

Map each caption to refined attributes, tagging images with Action/Background/Object.

Analysis (test.py)

Summarize subpopulation distributions and detect imbalances across dimensions.

Example Outputs

Refined Criteria JSON

json
{
  "Action": ["Surprise", "Curiosity", "Alarm", "Mystery"],
  "Background": ["grass", "beach", "trees", "wooden floor", "window", "autumn colors"],
  "Object": ["dog", "cat", "car", "ship"]
}
Image Assignment CSV

Code
filename,label,caption,Action,Background,Object
cifar_13.jpg,cat,"A blurry image of a cat...",,wooden floor,window,cat
cifar_17.jpg,dog,"A dog in a grassy field...",Curiosity,Mystery,grass,dog
Distribution Analysis

Code
--- Action Distribution ---
Surprise     3
Alarm        1
Curiosity    1
Mystery      1

Key Contributions

Designed a modular 5‑step pipeline for dataset analysis.

Applied LLMs as analysts to uncover latent subpopulations.

Produced structured metadata (criteria_refined.json, image_assignments.csv).

Demonstrated bias detection (e.g., overrepresentation of dogs outdoors).

Provided reproducible scripts for captioning, refinement, assignment, and analysis.

📂 Repository Structure
Code
lite_ssd/
│
├── step1_img_caption/
│   ├── captions/
│   │   └── captions_cifar10_qwen.csv
│   ├── demo.py
│   ├── infer_batch.py
│
├── step2_criteria_initialization.py
├── step3_criteria_refinement.py
├── step4_image_ass.py
├── test.py
├── criteria_refined.json
├── image_assignments.csv

Why This Project Matters

This project demonstrates how LLMs can bridge human reasoning and machine learning pipelines, surfacing hidden dataset structures that improve transparency, fairness, and evaluation. It’s a step toward LLM‑assisted dataset curation for real‑world AI systems.
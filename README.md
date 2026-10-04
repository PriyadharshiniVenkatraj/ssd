# Clinical Cohort Attribute Explorer

This project creates structured attributes from MIMIC-IV demo admission
diagnoses and provides a Streamlit dashboard for exploring the resulting
cohort. The dashboard supports filtering admissions, inspecting summaries,
and viewing condition, severity, and system distributions.

This is an exploratory research tool, not a clinical decision-support system.
Use only data you are authorized to access and follow the dataset's access and
usage terms.

## Workflow

1. `step1_img_caption/infer_batch.py` generates admission-level diagnosis
   summaries using the MIMIC-IV demo data and Qwen.
2. `step2_criteria_initialization.py` extracts candidate dimensions and
   attributes.
3. `step3_criteria_refinment.py` refines the attribute lists and writes
   `criteria_refined.json`.
4. `step4_image_ass.py` assigns criteria attributes to summaries and writes
   `image_assignments.csv`.
5. `streamlit_app.py` lets you explore the generated assignments.

## Run the Streamlit UI

From the project root, install the dependencies and start the dashboard:

```powershell
.\litessd\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app reads `image_assignments.csv` and `criteria_refined.json` from the
project root. If you update the criteria or source summaries, run
`step4_image_ass.py` to regenerate the assignments before launching the
dashboard.

## Dashboard pages

- **Overview** shows cohort counts and the most common condition, with
  distributions for each available dimension.
- **Admissions** provides a searchable, filterable table, admission summaries,
  and a CSV download of the filtered records.
- **Distributions** shows the most frequent attributes for a selected
  dimension.

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSIGNMENTS_PATH = ROOT / "image_assignments.csv"
CRITERIA_PATH = ROOT / "criteria_refined.json"
REQUIRED_COLUMNS = {"subject_id", "hadm_id", "base_caption"}


st.set_page_config(
    page_title="CohortLens | Clinical Cohort Explorer",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] {
        background: #f4f8f7;
        border: 1px solid #e3ece9;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    [data-testid="stMetricLabel"] { color: #52615d; }
    [data-testid="stMetricValue"] { color: #17332b; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    with CRITERIA_PATH.open(encoding="utf-8") as criteria_file:
        criteria = json.load(criteria_file)
    if not isinstance(criteria, dict) or not all(
        isinstance(dimension, str) and isinstance(attributes, list)
        for dimension, attributes in criteria.items()
    ):
        raise ValueError(f"Expected a JSON object of attribute lists in {CRITERIA_PATH}")

    data = pd.read_csv(ASSIGNMENTS_PATH)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {ASSIGNMENTS_PATH}: "
            f"{', '.join(sorted(missing))}"
        )
    missing_dimensions = set(criteria).difference(data.columns)
    if missing_dimensions:
        raise ValueError(
            f"Assignment data is missing criteria columns: "
            f"{', '.join(sorted(missing_dimensions))}. "
            "Run step4_image_ass.py to regenerate image_assignments.csv."
        )

    for column in ("subject_id", "hadm_id"):
        data[column] = (
            data[column].astype("string").str.replace(r"\.0$", "", regex=True).fillna("")
        )
    data["base_caption"] = data["base_caption"].fillna("").astype(str)
    return data, criteria


def split_tags(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [tag.strip() for tag in str(value).split(",") if tag.strip()]


def get_distribution(data: pd.DataFrame, dimension: str) -> pd.DataFrame:
    counts = (
        data[dimension]
        .map(split_tags)
        .explode()
        .dropna()
        .value_counts()
        .rename_axis("Attribute")
        .reset_index(name="Admissions")
    )
    if not counts.empty:
        counts["Share"] = (counts["Admissions"] / len(data) * 100).round(1)
    return counts


try:
    df, criteria = load_data()
except (OSError, json.JSONDecodeError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Could not load the cohort data: {error}")
    st.info(
        "Keep image_assignments.csv and criteria_refined.json in the project root, "
        "then regenerate the assignments if needed."
    )
    st.stop()

dimensions = list(criteria)
tag_columns = [column for column in dimensions if column in df.columns]

st.sidebar.title("CohortLens")
st.sidebar.caption("Clinical dataset explorer")
if st.sidebar.button("Reload generated data"):
    load_data.clear()
    st.rerun()
page = st.sidebar.radio("Workspace", ["Overview", "Admissions", "Distributions"])
st.sidebar.divider()
st.sidebar.subheader("Filter admissions")
search_text = st.sidebar.text_input(
    "Patient, admission, or diagnosis",
    placeholder="e.g. 10000032 or cirrhosis",
)

selected_tags: dict[str, list[str]] = {}
for dimension in tag_columns:
    available_tags = sorted(
        {
            tag
            for value in df[dimension]
            for tag in split_tags(value)
        },
        key=str.casefold,
    )
    selected_tags[dimension] = st.sidebar.multiselect(
        dimension,
        available_tags,
        placeholder=f"All {dimension.lower()} values",
    )

filtered = df.copy()
if search_text.strip():
    search_columns = ["subject_id", "hadm_id", "base_caption", *tag_columns]
    searchable = filtered[search_columns].fillna("").astype(str).agg(" ".join, axis=1)
    filtered = filtered[
        searchable.str.contains(search_text.strip(), case=False, regex=False)
    ]

for dimension, selected in selected_tags.items():
    if selected:
        selected_set = set(selected)
        filtered = filtered[
            filtered[dimension].map(
                lambda value: bool(selected_set.intersection(split_tags(value)))
            )
        ]

st.title("Clinical Cohort Explorer")
st.caption(
    "Explore admission-level clinical summaries and the structured attributes "
    "assigned by this project."
)

if filtered.empty:
    st.warning("No admissions match these filters. Try clearing one or more filters.")
    st.stop()

if page == "Overview":
    condition_counts = get_distribution(filtered, "Condition") if "Condition" in tag_columns else pd.DataFrame()
    most_common = (
        condition_counts.iloc[0]["Attribute"] if not condition_counts.empty else "—"
    )
    metrics = st.columns(3)
    metrics[0].metric("Admissions", f"{len(filtered):,}")
    metrics[1].metric("Unique patients", f"{filtered['subject_id'].nunique():,}")
    metrics[2].metric("Most common condition", most_common)

    st.subheader("Cohort at a glance")
    available_dimensions = [
        dimension
        for dimension in ("Condition", "Severity", "System")
        if dimension in tag_columns
    ]
    if available_dimensions:
        chart_columns = st.columns(len(available_dimensions))
        for column, dimension in zip(chart_columns, available_dimensions):
            with column:
                st.markdown(f"**{dimension}**")
                counts = get_distribution(filtered, dimension).head(8)
                if counts.empty:
                    st.caption("No assigned values in this cohort.")
                else:
                    st.bar_chart(counts.set_index("Attribute")["Admissions"])

    st.subheader("Recent records")
    preview_columns = ["subject_id", "hadm_id", *tag_columns]
    st.dataframe(
        filtered[preview_columns].head(10),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Use the Admissions page to inspect each admission's full summary.")

elif page == "Admissions":
    st.subheader("Admission records")
    st.write(f"{len(filtered):,} admissions match the current filters.")
    preview_columns = ["subject_id", "hadm_id", *tag_columns]
    st.dataframe(
        filtered[preview_columns],
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    st.download_button(
        "Download filtered admissions",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_admissions.csv",
        mime="text/csv",
    )

    st.markdown("#### Admission summary")
    record_indices = filtered.index.tolist()
    selected_index = st.selectbox(
        "Select an admission",
        record_indices,
        format_func=lambda index: (
            f"Admission {filtered.at[index, 'hadm_id']} · "
            f"Patient {filtered.at[index, 'subject_id']}"
        ),
    )
    selected_record = filtered.loc[selected_index]
    st.info(selected_record["base_caption"] or "No clinical summary is available.")
    summary_columns = st.columns(len(tag_columns)) if tag_columns else []
    for column, dimension in zip(summary_columns, tag_columns):
        with column:
            st.markdown(f"**{dimension}**")
            tags = split_tags(selected_record[dimension])
            st.write(", ".join(tags) if tags else "No attributes assigned")

elif page == "Distributions":
    st.subheader("Attribute distributions")
    if not tag_columns:
        st.info("No criteria dimensions are available in the assignment data.")
    else:
        dimension = st.selectbox("Dimension", tag_columns)
        counts = get_distribution(filtered, dimension)
        if counts.empty:
            st.info(f"No {dimension.lower()} attributes are assigned in this cohort.")
        else:
            top_n = st.slider(
                "Attributes to show",
                min_value=1,
                max_value=min(30, len(counts)),
                value=min(12, len(counts)),
            )
            visible_counts = counts.head(top_n)
            st.bar_chart(visible_counts.set_index("Attribute")["Admissions"])
            st.dataframe(
                visible_counts,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Share": st.column_config.NumberColumn(format="%.1f%%")
                },
            )
            st.caption(
                "An admission can have multiple attributes in the same dimension, "
                "so percentages may add up to more than 100%."
            )

st.divider()
st.caption(
    "For exploratory research only—not for diagnosis or treatment. "
    "Use de-identified data and follow your dataset's access and usage terms."
)

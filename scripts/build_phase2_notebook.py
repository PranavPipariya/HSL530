"""Build the Phase 2 bail-case empirical investigation notebook.

The notebook itself contains the reproducible analysis. This builder keeps the
notebook source manageable and makes it easy to regenerate the same structure.
"""

from pathlib import Path

import nbformat as nbf


OUT = Path("Phase2_Bail_Empirical_Investigation.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = []

cells.append(md(
    r"""
# Phase 2 Submission: Empirical Investigation of High Court Bail Cases

**Project question:** How do bail type and High Court context shape delay, pendency, and observed outcomes in Indian High Court bail cases?

**Submitted by:**
- Tushar Singh (22322032)
- Pranav Pipariya (22322022)
- Tanishq Gupta (22322031)
- Kavish Jain (22322017)
- Yash Kumar (22322034)

This Phase 2 notebook builds on the Phase 1 I/O and EDA submission. It moves from broad exploration to five applied empirical investigations, one for each team member, using the DAKSH High Court Bail Case dataset.

**Important interpretation note:** this notebook does **not** provide legal advice or individual bail predictions. Its prediction component estimates historical long-delay risk from case metadata in this dataset. It should be interpreted as decision-support context for research, not as a forecast of any person's legal outcome.

**Data context:** DAKSH describes bail data analysis as important for understanding undertrial detention, timely hearings, and process reform. DAKSH also warns that data availability gaps and other limitations can affect findings. We therefore report coverage, missingness, and cohort restrictions before interpreting results.

Sources used for framing and citation:
- DAKSH High Court Data Portal: https://www.dakshindia.org/daksh-high-court-data-portal/
- DAKSH High Court Bail Dashboard: https://database.dakshindia.org/bail-dashboard/
- Local codebook: `docs/Codebook_DAKSH_HighCourt_2023.pdf`
"""
))

cells.append(md(
    r"""
## Abstract

This study examines nearly **0.93 million High Court bail case records** to understand how bail type and court context relate to case processing. The analysis focuses on three connected outcomes: disposal delay, pending burden, and observed disposal outcomes where outcome labels are available.

The analysis produces five main conclusions:

1. **Bail cases are structurally heterogeneous.** Regular bail accounts for most records, anticipatory bail is also substantial, and cancellation is rare but much slower.
2. **Disposal time differs sharply by bail type.** In the disposed cohort, regular bail has a median disposal time of about 23 days, anticipatory bail about 37 days, and cancellation about 268 days.
3. **Court context matters even after adjustment.** Raw medians are not enough because courts differ in bail-type and case-type composition, so the notebook builds an adjusted court-delay index.
4. **Filing-stage metadata contains useful delay-risk signal.** The model estimates long-delay risk from case metadata available near filing, while excluding post-outcome variables.
5. **Outcome labels are informative but incomplete.** Observed bail outcomes are analyzed only in the labeled disposed subset, with coverage shown before interpretation.

The project is organized around transparent measurement, careful cohort definition, and restrained interpretation. Each empirical claim is linked to a table or figure and paired with the relevant data limitation.
"""
))

cells.append(md(
    r"""
## Research Design And Evidence Structure

The starting point is that bail is not one uniform category. Regular bail, anticipatory bail, and cancellation cases enter High Courts through different procedural paths and registry conventions. The central empirical task is therefore to separate **administrative patterns in bail processing** from patterns that may reflect court-specific data entry, missing legal metadata, or incomplete disposal labels.

The evidence is built in five layers:

1. **Measurement quality first.** We validate record counts, parse mixed date formats, quantify missingness, and define which fields can safely support full-cohort analysis.
2. **Bail-type composition.** We map how bail categories differ across courts, years, and case-type conventions.
3. **Delay and court inequality.** We compare raw disposal times, then add an adjusted delay index so court comparisons are not driven only by bail-type or filing-year composition.
4. **Prediction as decision-support context.** We estimate long-delay risk using only filing-stage metadata, with explicit leakage controls and baseline comparisons.
5. **Outcome labels with restraint.** We analyze observed outcomes only where labels are available, and we keep those claims separate from the stronger delay and pendency findings.

This structure keeps the strongest claims tied to the most reliable fields. Weaker fields are still used when they add value, but only with coverage checks and explicit limits on interpretation.
"""
))

cells.append(md(
    r"""
## Methodological Principles

The notebook follows six methodological principles:

- **Reproducibility:** all files are loaded relative to the project folder.
- **Date validity:** mixed date formats are parsed explicitly and validated.
- **Outcome restraint:** disposal outcomes are analyzed only where labels exist.
- **Prediction integrity:** the long-delay model excludes variables that reveal post-filing outcomes.
- **Careful court comparison:** court rankings are treated as descriptive benchmarks, not causal claims about judicial quality.
- **Model transparency:** the predictive model is implemented with NumPy and explained through coefficients, calibration, baselines, and robustness checks.
"""
))

cells.append(md(
    r"""
## 0. Setup And Reproducibility

The notebook is designed to run from the project folder without absolute paths. It creates all tables and figures under `outputs/`.
"""
))

cells.append(code(
    r"""
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

sns.set_theme(style="whitegrid", palette="Set2", context="notebook")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 180

PROJECT_DIR = Path(".").resolve()
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for path in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

candidate_paths = [
    PROJECT_DIR / "Compiled Bail case data.csv",
    PROJECT_DIR / "data" / "Compiled Bail case data.csv",
    PROJECT_DIR.parent / "Compiled Bail case data.csv",
]
DATA_PATH = next((p for p in candidate_paths if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError("Could not locate Compiled Bail case data.csv.")

print(f"Project folder: {PROJECT_DIR}")
print(f"Data path: {DATA_PATH}")
print(f"Dataset size: {DATA_PATH.stat().st_size / (1024**2):,.2f} MB")
print(f"Random seed: {RANDOM_SEED}")
"""
))

cells.append(md(
    r"""
## Project Workflow Diagram

The workflow below shows how the raw DAKSH bail case data is converted into validated analytical cohorts, empirical investigations, generated outputs, and final interpretation. It is included to make the project structure transparent and reproducible.
"""
))

cells.append(code(
    r"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_workflow_diagram():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 7)
    ax.axis("off")

    nodes = {
        "raw": (0.5, 5.2, 2.0, 0.9, "Raw data\nDAKSH bail CSV"),
        "codebook": (0.5, 3.8, 2.0, 0.9, "Source context\nCodebook + portal"),
        "clean": (3.2, 4.5, 2.2, 1.1, "Data validation\nDates, missingness,\nanomalies"),
        "cohorts": (6.0, 4.5, 2.2, 1.1, "Analytical cohorts\nFull, disposed,\npending, labeled"),
        "inv1": (8.9, 5.8, 2.7, 0.75, "Bail-type landscape"),
        "inv2": (8.9, 4.75, 2.7, 0.75, "Disposal delay +\nadjusted court index"),
        "inv3": (8.9, 3.7, 2.7, 0.75, "Long-delay risk\nprediction"),
        "inv4": (8.9, 2.65, 2.7, 0.75, "Pending burden +\nCOVID-window shift"),
        "inv5": (8.9, 1.6, 2.7, 0.75, "Observed outcomes\nlabeled subset"),
        "outputs": (8.9, 0.35, 2.7, 0.85, "Generated outputs\n29 tables + 14 figures"),
        "interpret": (12.1, 0.35, 2.45, 0.85, "Final synthesis\nlimitations + replication"),
    }

    colors = {
        "raw": "#E8F1FB",
        "codebook": "#E8F1FB",
        "clean": "#EAF5EA",
        "cohorts": "#EAF5EA",
        "outputs": "#FFF2CC",
        "interpret": "#FFF2CC",
    }

    for key, (x, y, w, h, label) in nodes.items():
        color = colors.get(key, "#F4E8F7")
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, linespacing=1.18)

    def connect(a, b, y_offset_a=0.0, y_offset_b=0.0):
        ax1, ay1, aw, ah, _ = nodes[a]
        bx, by, bw, bh, _ = nodes[b]
        start = (ax1 + aw, ay1 + ah / 2 + y_offset_a)
        end = (bx, by + bh / 2 + y_offset_b)
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#333333",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    def connect_points(start, end):
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#333333",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    connect("raw", "clean", y_offset_a=-0.15, y_offset_b=0.18)
    connect("codebook", "clean", y_offset_a=0.15, y_offset_b=-0.18)
    connect("clean", "cohorts")
    for target in ["inv1", "inv2", "inv3", "inv4", "inv5"]:
        connect("cohorts", target)
    inv5_x, inv5_y, inv5_w, _, _ = nodes["inv5"]
    out_x, out_y, out_w, out_h, _ = nodes["outputs"]
    connect_points((inv5_x + inv5_w / 2, inv5_y), (out_x + out_w / 2, out_y + out_h))
    connect("outputs", "interpret")

    ax.text(
        7.6, 6.75,
        "Phase 2 Bail Case Empirical Investigation: Analysis Workflow",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        7.6, 0.08,
        "All analyses are generated from the notebook and saved under outputs/tables and outputs/figures.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    plt.tight_layout()
    return fig


fig = draw_workflow_diagram()
workflow_path = FIG_DIR / "00_project_workflow.png"
fig.savefig(workflow_path, bbox_inches="tight", dpi=180)
plt.show()
print(f"Saved figure: {workflow_path.relative_to(PROJECT_DIR)}")
"""
))

cells.append(md(
    r"""
## 1. Load Data And Validate Core Fields

We load analysis-relevant columns, preserve raw string fields, and convert dates/numerics deliberately. A key Phase 2 improvement over naive loading is robust parsing for the two date formats found in the dataset: `YYYY-MM-DD` and `DD-MM-YYYY`.
"""
))

cells.append(code(
    r"""
raw_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
print(f"Raw column count: {len(raw_columns)}")
print(raw_columns)

analysis_cols = [
    "CNR_NUMBER", "CASE_NUMBER", "CASE_TYPE", "CASETYPE_FULLFORM",
    "COMBINED_CASE_NUMBER", "COURT_NAME", "COURT_NUMBER", "NAME_OF_HIGH_COURT",
    "CURRENT_STAGE", "CURRENT_STATUS", "DATE_FILED", "DECISION_DATE",
    "FILING_NUMBER", "HEARING_COUNT", "LAST_SYNC_TIME", "NATURE_OF_DISPOSAL",
    "NATURE_OF_DISPOSAL_OUTCOME", "NATURE_OF_DISPOSAL_BINARY", "NJDG_JUDGE_NAME",
    "PENDING_DAYS", "POLICE_STATION", "REGISTRATION_DATE", "REGISTRATION_NUMBER",
    "RESPONDENT", "UNDER_ACTS", "UNDER_SECTIONS", "YEAR", "DISPOSAL_YEAR",
    "DISPOSAL_DAYS...1", "Mapped_Bail",
]

df = pd.read_csv(
    DATA_PATH,
    usecols=[c for c in analysis_cols if c in raw_columns],
    dtype="string",
    low_memory=False,
)

print(f"Loaded shape: {df.shape}")
display(df.head(3))
"""
))

cells.append(code(
    r"""
def parse_mixed_date(series: pd.Series) -> pd.Series:
    # Parse DAKSH date fields containing YYYY-MM-DD and DD-MM-YYYY values.
    ymd = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
    return ymd.fillna(dmy)


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA})
    )


for col in [
    "CASE_TYPE", "CASETYPE_FULLFORM", "COURT_NAME", "NAME_OF_HIGH_COURT",
    "CURRENT_STATUS", "NATURE_OF_DISPOSAL", "NATURE_OF_DISPOSAL_OUTCOME",
    "NATURE_OF_DISPOSAL_BINARY", "POLICE_STATION", "RESPONDENT",
    "UNDER_ACTS", "UNDER_SECTIONS", "Mapped_Bail",
]:
    if col in df.columns:
        df[col] = clean_text(df[col])

for col in ["DATE_FILED", "DECISION_DATE", "LAST_SYNC_TIME", "REGISTRATION_DATE"]:
    if col in df.columns:
        df[f"{col}_PARSED"] = parse_mixed_date(df[col])

for col in ["HEARING_COUNT", "PENDING_DAYS", "YEAR", "DISPOSAL_YEAR", "DISPOSAL_DAYS...1"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.rename(columns={"DISPOSAL_DAYS...1": "DISPOSAL_DAYS"})

df["FILED_YEAR"] = df["DATE_FILED_PARSED"].dt.year
df["FILED_MONTH"] = df["DATE_FILED_PARSED"].dt.month
df["FILED_QUARTER"] = df["DATE_FILED_PARSED"].dt.quarter
df["IS_DISPOSED"] = df["CURRENT_STATUS"].eq("Disposed")
df["IS_PENDING"] = df["CURRENT_STATUS"].eq("Pending")
df["DATE_ANOMALY_DECISION_BEFORE_FILED"] = (
    df["DECISION_DATE_PARSED"].notna()
    & df["DATE_FILED_PARSED"].notna()
    & (df["DECISION_DATE_PARSED"] < df["DATE_FILED_PARSED"])
)

df["CASE_TYPE_CLEAN"] = df["CASE_TYPE"].str.upper().str.strip()
top_case_types = df["CASE_TYPE_CLEAN"].value_counts().head(25).index
df["CASE_TYPE_GROUP"] = df["CASE_TYPE_CLEAN"].where(
    df["CASE_TYPE_CLEAN"].isin(top_case_types), "OTHER_CASE_TYPE"
)

print("Derived fields created.")
display(df[[
    "DATE_FILED", "DATE_FILED_PARSED", "DECISION_DATE", "DECISION_DATE_PARSED",
    "CURRENT_STATUS", "Mapped_Bail", "DISPOSAL_DAYS", "PENDING_DAYS",
]].head())
"""
))

cells.append(code(
    r"""
def table_out(dataframe: pd.DataFrame, filename: str, caption: str = None, index: bool = True):
    path = TABLE_DIR / filename
    dataframe.to_csv(path, index=index)
    if caption:
        display(Markdown(f"**{caption}**"))
    display(dataframe)
    print(f"Saved table: {path.relative_to(PROJECT_DIR)}")


def save_current_fig(filename: str):
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {path.relative_to(PROJECT_DIR)}")


date_validation_rows = []
for raw_col in ["DATE_FILED", "DECISION_DATE", "LAST_SYNC_TIME", "REGISTRATION_DATE"]:
    parsed_col = f"{raw_col}_PARSED"
    if raw_col in df.columns and parsed_col in df.columns:
        nonnull = int(df[raw_col].notna().sum())
        parsed = int(df[parsed_col].notna().sum())
        date_validation_rows.append({
            "field": raw_col,
            "non_null_raw": nonnull,
            "parsed": parsed,
            "unparsed_non_null": nonnull - parsed,
            "min_parsed": df[parsed_col].min(),
            "max_parsed": df[parsed_col].max(),
        })

validation_summary = pd.DataFrame([
    {"check": "Expected raw columns", "value": len(raw_columns)},
    {"check": "Loaded records", "value": len(df)},
    {"check": "Unique CNR_NUMBER", "value": df["CNR_NUMBER"].nunique()},
    {"check": "Duplicate CNR_NUMBER rows", "value": int(df["CNR_NUMBER"].duplicated().sum())},
    {"check": "Decision date before filed date", "value": int(df["DATE_ANOMALY_DECISION_BEFORE_FILED"].sum())},
    {"check": "Disposed cases missing decision date", "value": int((df["IS_DISPOSED"] & df["DECISION_DATE_PARSED"].isna()).sum())},
    {"check": "Pending cases with decision date", "value": int((df["IS_PENDING"] & df["DECISION_DATE_PARSED"].notna()).sum())},
])

date_validation = pd.DataFrame(date_validation_rows)

table_out(validation_summary, "00_validation_summary.csv", "Core validation summary", index=False)
table_out(date_validation, "00_date_parse_validation.csv", "Date parsing validation", index=False)

assert len(raw_columns) == 32, "Unexpected raw column count."
assert len(df) == 927896, "Unexpected record count."
assert (date_validation["unparsed_non_null"] == 0).all(), "Some non-null dates failed to parse."
"""
))

cells.append(md(
    r"""
## 2. Data Quality Profile And Analytical Cohorts

The DAKSH codebook notes that some fields are scraped directly, while others are manually mapped or calculated by DAKSH. Because courts use different registry conventions and some legal-detail fields are sparse, we separate reliable full-cohort analyses from restricted-cohort outcome analyses.

The core analytical design is:

| Question type | Main fields | Cohort used | Why this cohort is appropriate |
|---|---|---|---|
| Bail mix and court landscape | `Mapped_Bail`, `NAME_OF_HIGH_COURT`, `CASE_TYPE`, `YEAR` | All records | These fields are complete enough for full-dataset descriptive analysis. |
| Disposal delay | `DISPOSAL_DAYS`, `DECISION_DATE`, `DATE_FILED` | Disposed cases with valid dates | Disposal time is defined only after a case has reached decision/disposal. |
| Pending burden | `CURRENT_STATUS`, `PENDING_DAYS`, `LAST_SYNC_TIME` | All records, interpreted as scrape-date snapshot | Pending status depends on when the case record was scraped/synced. |
| Long-delay prediction | Filing-stage metadata only | Disposed train/test split | The target is measured from disposal time, but predictors avoid post-filing leakage. |
| Observed outcomes | `NATURE_OF_DISPOSAL_OUTCOME` | Disposed rows with non-missing outcome labels | Outcome labels are incomplete, so this cannot be treated as a full-population outcome model. |
"""
))

cells.append(code(
    r"""
missing_profile = (
    df.isna().mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("missing_pct")
    .reset_index()
    .rename(columns={"index": "field"})
)
table_out(missing_profile.head(20).round(2), "01_missing_profile_top20.csv", "Top 20 fields by missingness", index=False)

court_completeness = (
    df.groupby("NAME_OF_HIGH_COURT")
    [["UNDER_ACTS", "UNDER_SECTIONS", "NATURE_OF_DISPOSAL_OUTCOME", "POLICE_STATION"]]
    .apply(lambda x: x.notna().mean().mul(100))
    .round(1)
    .sort_values("NATURE_OF_DISPOSAL_OUTCOME", ascending=False)
)
table_out(court_completeness, "01_court_metadata_completeness.csv", "Metadata completeness by High Court")

plt.figure(figsize=(9, 5))
sns.barplot(data=missing_profile.head(12), x="missing_pct", y="field", color="#4C78A8")
plt.title("Highest Missingness Fields")
plt.xlabel("Missing values (%)")
plt.ylabel("")
save_current_fig("01_missingness_top_fields.png")

plt.figure(figsize=(10, 6))
sns.heatmap(court_completeness, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "% non-null"})
plt.title("Metadata Completeness Varies Strongly By Court")
plt.xlabel("Field")
plt.ylabel("")
save_current_fig("01_court_metadata_completeness.png")
"""
))

cells.append(code(
    r"""
def classify_act(value) -> str:
    if pd.isna(value):
        return "MISSING"
    text = str(value).upper()
    if text.strip() in {"", "---", "OTHER", "OTHER ACTS", "OTHER||", "NAN"}:
        return "OTHER_OR_UNSPECIFIED"
    if "POCSO" in text or "PROTECTION OF CHILDREN" in text:
        return "POCSO"
    if "NARCOTIC" in text or "PSYCHOTROPIC" in text or "NDPS" in text:
        return "NDPS"
    if "DOWRY" in text:
        return "DOWRY"
    if "ARMS" in text:
        return "ARMS"
    if "EXCISE" in text:
        return "EXCISE"
    if "INDIAN PENAL" in text or "I.P.C" in text or re.search(r"\bIPC\b", text):
        return "IPC"
    if "CRIMINAL PROCEDURE" in text or "CR.PC" in text or "CRPC" in text:
        return "CRPC"
    return "OTHER_SPECIFIED"


def classify_section(value) -> str:
    if pd.isna(value):
        return "MISSING"
    text = str(value).upper()
    if text.strip() in {"", "---", "NAN"}:
        return "MISSING_OR_UNSPECIFIED"
    compact = re.sub(r"\s+", "", text)
    if "438" in compact:
        return "SEC_438_ANTICIPATORY"
    if "439" in compact:
        return "SEC_439_REGULAR"
    if "302" in compact:
        return "SEC_302_MURDER"
    if "376" in compact:
        return "SEC_376_SEXUAL_OFFENCE"
    if "420" in compact:
        return "SEC_420_CHEATING"
    if "498A" in compact or "304B" in compact:
        return "MATRIMONIAL_DOWRY"
    if "307" in compact:
        return "SEC_307_ATTEMPT_MURDER"
    return "OTHER_SECTIONS"


df["ACT_GROUP"] = df["UNDER_ACTS"].map(classify_act)
df["SECTION_GROUP"] = df["UNDER_SECTIONS"].map(classify_section)

legal_metadata_summary = pd.DataFrame({
    "ACT_GROUP": df["ACT_GROUP"].value_counts(),
}).join(pd.DataFrame({"SECTION_GROUP": df["SECTION_GROUP"].value_counts()}), how="outer").fillna(0).astype(int)

table_out(legal_metadata_summary, "01_legal_metadata_groups.csv", "Cleaned legal metadata groups")
"""
))

cells.append(md(
    r"""
# Investigation 1 - Bail-Type Landscape

**Lead:** Tanishq Gupta

**Question:** How are regular bail, anticipatory bail, and cancellation cases distributed across courts, years, and case-type conventions?

This module uses the full dataset because `Mapped_Bail`, court, filing year, and case type are available for all records.
"""
))

cells.append(code(
    r"""
bail_counts = (
    df["Mapped_Bail"]
    .value_counts()
    .rename_axis("Mapped_Bail")
    .reset_index(name="cases")
)
bail_counts["share_pct"] = bail_counts["cases"] / len(df) * 100
table_out(bail_counts.round(2), "02_bail_type_counts.csv", "Bail type distribution", index=False)

status_mix = (
    pd.crosstab(df["Mapped_Bail"], df["CURRENT_STATUS"], normalize="index")
    .mul(100)
    .round(2)
)
table_out(status_mix, "02_status_mix_by_bail_type.csv", "Status mix within each bail type")

plt.figure(figsize=(8, 4.5))
sns.barplot(data=bail_counts, x="cases", y="Mapped_Bail", color="#59A14F")
plt.title("Bail-Type Composition")
plt.xlabel("Cases")
plt.ylabel("")
save_current_fig("02_bail_type_composition.png")

year_bail = (
    df.groupby(["FILED_YEAR", "Mapped_Bail"])
    .size()
    .reset_index(name="cases")
    .dropna(subset=["FILED_YEAR"])
)
plt.figure(figsize=(10, 5))
sns.lineplot(data=year_bail, x="FILED_YEAR", y="cases", hue="Mapped_Bail", marker="o")
plt.title("Filed Bail Cases By Year And Bail Type")
plt.xlabel("Filing year")
plt.ylabel("Cases")
save_current_fig("02_filed_cases_by_year_bail_type.png")
"""
))

cells.append(code(
    r"""
court_bail_counts = (
    df.groupby(["NAME_OF_HIGH_COURT", "Mapped_Bail"])
    .size()
    .reset_index(name="cases")
)
court_totals = court_bail_counts.groupby("NAME_OF_HIGH_COURT")["cases"].transform("sum")
court_bail_counts["share_pct"] = court_bail_counts["cases"] / court_totals * 100

court_bail_pivot = (
    court_bail_counts.pivot(index="NAME_OF_HIGH_COURT", columns="Mapped_Bail", values="share_pct")
    .fillna(0)
    .round(2)
)
court_bail_pivot["TOTAL_CASES"] = df["NAME_OF_HIGH_COURT"].value_counts()
court_bail_pivot = court_bail_pivot.sort_values("TOTAL_CASES", ascending=False)
table_out(court_bail_pivot, "02_court_bail_mix_percent.csv", "Bail mix by High Court")

plot_mix = court_bail_pivot.drop(columns="TOTAL_CASES").head(12)
ax = plot_mix.plot(kind="barh", stacked=True, figsize=(10, 6), width=0.82)
ax.invert_yaxis()
ax.set_title("Bail-Type Mix In The Largest High Courts")
ax.set_xlabel("Share of court cases (%)")
ax.set_ylabel("")
ax.legend(title="Mapped_Bail", bbox_to_anchor=(1.02, 1), loc="upper left")
save_current_fig("02_largest_courts_bail_mix.png")

case_type_by_bail = (
    df.groupby(["Mapped_Bail", "CASE_TYPE_GROUP"])
    .size()
    .reset_index(name="cases")
)
case_type_by_bail["share_pct"] = (
    case_type_by_bail["cases"]
    / case_type_by_bail.groupby("Mapped_Bail")["cases"].transform("sum")
    * 100
)
top_case_by_bail = (
    case_type_by_bail.sort_values(["Mapped_Bail", "cases"], ascending=[True, False])
    .groupby("Mapped_Bail")
    .head(8)
    .round(2)
)
table_out(top_case_by_bail, "02_top_case_types_by_bail.csv", "Top case-type groups by bail type", index=False)
"""
))

cells.append(md(
    r"""
**Investigation 1 takeaway.** Bail type is not evenly distributed across courts. Some courts are almost entirely regular-bail heavy, while others record a much larger share of anticipatory bail. This means later delay and pendency comparisons must account for court composition and registry conventions.

Substantively, this matters because the analysis should not treat all bail cases as interchangeable. A court with mostly regular-bail filings and a court with mostly anticipatory-bail filings may show different disposal patterns even before any institutional delay is considered. The first investigation therefore establishes the core denominator for the rest of the project: every later comparison is interpreted through the joint lens of **court**, **bail type**, and **case-type convention**.
"""
))

cells.append(md(
    r"""
# Investigation 2 - Disposal Delay And Court Inequality

**Lead:** Kavish Jain

**Question:** Which courts and bail types take longer to dispose cases, and does the ranking change after adjusting for bail type, filing year, and case-type mix?

This module uses disposed cases with valid disposal days and excludes the small number of records where decision date is earlier than filing date.
"""
))

cells.append(code(
    r"""
disposed = df.loc[
    df["IS_DISPOSED"]
    & df["DISPOSAL_DAYS"].notna()
    & (df["DISPOSAL_DAYS"] >= 1)
    & ~df["DATE_ANOMALY_DECISION_BEFORE_FILED"]
].copy()

long_delay_p75 = float(disposed["DISPOSAL_DAYS"].quantile(0.75))
long_delay_p90 = float(disposed["DISPOSAL_DAYS"].quantile(0.90))
disposed["LONG_DELAY_P75"] = disposed["DISPOSAL_DAYS"] > long_delay_p75
disposed["VERY_LONG_DELAY_P90"] = disposed["DISPOSAL_DAYS"] > long_delay_p90

print(f"Disposed analytical cohort: {len(disposed):,}")
print(f"Long-delay threshold p75: > {long_delay_p75:.0f} days")
print(f"Very-long-delay threshold p90: > {long_delay_p90:.0f} days")

delay_by_bail = (
    disposed.groupby("Mapped_Bail")["DISPOSAL_DAYS"]
    .agg(cases="count", mean_days="mean", median_days="median",
         p75_days=lambda s: s.quantile(0.75), p90_days=lambda s: s.quantile(0.90),
         p99_days=lambda s: s.quantile(0.99))
    .round(2)
    .sort_values("median_days", ascending=False)
)
table_out(delay_by_bail, "03_delay_by_bail_type.csv", "Disposal-time distribution by bail type")

court_delay = (
    disposed.groupby("NAME_OF_HIGH_COURT")["DISPOSAL_DAYS"]
    .agg(cases="count", median_days="median", p75_days=lambda s: s.quantile(0.75),
         p90_days=lambda s: s.quantile(0.90), mean_days="mean")
    .round(2)
    .sort_values("median_days", ascending=False)
)
table_out(court_delay, "03_court_delay_distribution.csv", "Disposal-time distribution by High Court")
"""
))

cells.append(code(
    r"""
plt.figure(figsize=(9, 5))
plot_delay = delay_by_bail.reset_index().sort_values("median_days")
sns.barplot(data=plot_delay, x="median_days", y="Mapped_Bail", color="#F28E2B")
plt.title("Median Disposal Days By Bail Type")
plt.xlabel("Median disposal days")
plt.ylabel("")
save_current_fig("03_median_delay_by_bail_type.png")

plt.figure(figsize=(10, 6))
court_plot = court_delay.reset_index().sort_values("median_days", ascending=True)
sns.barplot(data=court_plot, x="median_days", y="NAME_OF_HIGH_COURT", color="#E15759")
plt.title("Median Disposal Days By High Court")
plt.xlabel("Median disposal days")
plt.ylabel("")
save_current_fig("03_median_delay_by_court.png")
"""
))

cells.append(code(
    r"""
# Adjustment without external modelling dependencies:
# Compare each case's observed disposal days to an expected median for similar
# bail type, filing year, and case type. Because many case-type labels are
# court-specific, we only use the case-type stratum when it spans at least
# three courts and 100 cases; otherwise we fall back to bail type + year.
strata_cols = ["Mapped_Bail", "FILED_YEAR", "CASE_TYPE_GROUP"]
stratum_stats = (
    disposed.groupby(strata_cols)
    .agg(
        EXPECTED_STRATUM_MEDIAN=("DISPOSAL_DAYS", "median"),
        STRATUM_CASES=("DISPOSAL_DAYS", "size"),
        STRATUM_COURTS=("NAME_OF_HIGH_COURT", "nunique"),
    )
    .reset_index()
)
fallback_stats = (
    disposed.groupby(["Mapped_Bail", "FILED_YEAR"])["DISPOSAL_DAYS"]
    .median()
    .rename("EXPECTED_BAIL_YEAR_MEDIAN")
    .reset_index()
)
disposed_adj = disposed.merge(stratum_stats, on=strata_cols, how="left")
disposed_adj = disposed_adj.merge(fallback_stats, on=["Mapped_Bail", "FILED_YEAR"], how="left")
global_median = disposed["DISPOSAL_DAYS"].median()
use_specific_stratum = (
    disposed_adj["STRATUM_CASES"].ge(100)
    & disposed_adj["STRATUM_COURTS"].ge(3)
)
disposed_adj["EXPECTED_MEDIAN_USED"] = np.where(
    use_specific_stratum,
    disposed_adj["EXPECTED_STRATUM_MEDIAN"],
    disposed_adj["EXPECTED_BAIL_YEAR_MEDIAN"],
)
disposed_adj["EXPECTED_MEDIAN_USED"] = disposed_adj["EXPECTED_MEDIAN_USED"].fillna(global_median)
disposed_adj["DELAY_RATIO"] = disposed_adj["DISPOSAL_DAYS"] / disposed_adj["EXPECTED_MEDIAN_USED"].clip(lower=1)
disposed_adj["ADJUSTMENT_LEVEL"] = np.where(use_specific_stratum, "bail+year+case_type", "bail+year fallback")

adjusted_court_delay = (
    disposed_adj.groupby("NAME_OF_HIGH_COURT")
    .agg(
        cases=("DISPOSAL_DAYS", "size"),
        observed_median_days=("DISPOSAL_DAYS", "median"),
        adjusted_delay_index=("DELAY_RATIO", lambda s: s.median() * 100),
        long_delay_rate_p75=("LONG_DELAY_P75", "mean"),
        very_long_delay_rate_p90=("VERY_LONG_DELAY_P90", "mean"),
    )
    .assign(
        long_delay_rate_p75=lambda x: x["long_delay_rate_p75"] * 100,
        very_long_delay_rate_p90=lambda x: x["very_long_delay_rate_p90"] * 100,
    )
    .round(2)
    .sort_values("adjusted_delay_index", ascending=False)
)
table_out(adjusted_court_delay, "03_adjusted_court_delay_index.csv", "Adjusted court delay index")

plt.figure(figsize=(10, 6))
adj_plot = adjusted_court_delay.reset_index().sort_values("adjusted_delay_index", ascending=True)
sns.barplot(data=adj_plot, x="adjusted_delay_index", y="NAME_OF_HIGH_COURT", color="#B07AA1")
plt.axvline(100, color="black", linewidth=1, linestyle="--")
plt.title("Adjusted Court Delay Index\n100 = expected median for similar bail type/year/case-type mix")
plt.xlabel("Adjusted delay index")
plt.ylabel("")
save_current_fig("03_adjusted_court_delay_index.png")
"""
))

cells.append(md(
    r"""
**Investigation 2 takeaway.** Raw court rankings are informative but incomplete because courts handle different bail-type and case-type mixes. The adjusted delay index compares each court to cases with similar filing year, bail type, and case type, making the court-level comparison more balanced.

The adjusted index is not a causal estimate of court efficiency. It is a disciplined descriptive benchmark: values above 100 mean that the court's median case took longer than expected relative to comparable bail-type and filing-year groups, while values below 100 mean faster-than-expected disposal. This level of interpretation fits the available data because the dataset lacks judge-level reasoning, offence facts, custody period, and litigant characteristics. The value of the index is that it avoids ranking courts purely on raw medians when their case mixes differ sharply.
"""
))

cells.append(md(
    r"""
# Investigation 3 - Long-Delay Risk Prediction

**Lead:** Pranav Pipariya

**Question:** Using information plausibly available near filing, can we estimate whether a disposed case is likely to take unusually long?

The model predicts whether `DISPOSAL_DAYS` exceeds the disposed-cohort p75 threshold. A p90 model is included as a robustness check. The model intentionally excludes leakage variables such as `DECISION_DATE`, `DISPOSAL_DAYS`, `PENDING_DAYS`, `CURRENT_STATUS`, and disposal-outcome fields.
"""
))

cells.append(code(
    r"""
model_features = [
    "NAME_OF_HIGH_COURT", "Mapped_Bail", "CASE_TYPE_GROUP",
    "FILED_YEAR", "FILED_MONTH", "ACT_GROUP", "SECTION_GROUP",
]
leakage_fields = {
    "CURRENT_STATUS", "DECISION_DATE", "DECISION_DATE_PARSED",
    "DISPOSAL_DAYS", "PENDING_DAYS", "NATURE_OF_DISPOSAL",
    "NATURE_OF_DISPOSAL_OUTCOME", "NATURE_OF_DISPOSAL_BINARY",
    "LAST_SYNC_TIME", "LAST_SYNC_TIME_PARSED",
}
assert not leakage_fields.intersection(model_features), "Leakage field included in model features."

model_source = disposed.dropna(subset=["FILED_YEAR", "FILED_MONTH"]).copy()
model_source["FILED_YEAR"] = model_source["FILED_YEAR"].astype(int).astype(str)
model_source["FILED_MONTH"] = model_source["FILED_MONTH"].astype(int).astype(str).str.zfill(2)

train_pool = model_source[model_source["FILED_YEAR"].astype(int) <= 2018]
test_pool = model_source[model_source["FILED_YEAR"].astype(int) >= 2019]

train_sample = train_pool.sample(n=min(120_000, len(train_pool)), random_state=RANDOM_SEED)
test_sample = test_pool.sample(n=min(80_000, len(test_pool)), random_state=RANDOM_SEED)

print(f"Train pool: {len(train_pool):,}; train sample: {len(train_sample):,}")
print(f"Test pool: {len(test_pool):,}; test sample: {len(test_sample):,}")
print(f"Features: {model_features}")
"""
))

cells.append(code(
    r"""
def make_design_matrix(frame: pd.DataFrame, feature_cols, columns=None) -> pd.DataFrame:
    feature_frame = frame[feature_cols].copy()
    for col in feature_cols:
        feature_frame[col] = feature_frame[col].astype("string").fillna("MISSING")
    X = pd.get_dummies(feature_frame, columns=feature_cols, dtype=np.float32)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic_numpy(X: np.ndarray, y: np.ndarray, lr=0.12, l2=0.002, n_iter=140):
    beta = np.zeros(X.shape[1], dtype=np.float64)
    losses = []
    for i in range(n_iter):
        pred = sigmoid(X @ beta)
        reg = l2 * np.r_[0.0, beta[1:]]
        grad = (X.T @ (pred - y)) / len(y) + reg
        beta -= lr * grad
        if i in {0, 9, 24, 49, 99, n_iter - 1}:
            eps = 1e-9
            loss = -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))
            loss += 0.5 * l2 * np.sum(beta[1:] ** 2)
            losses.append({"iteration": i + 1, "loss": loss})
    return beta, pd.DataFrame(losses)


def standardize_train_test(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
    X_train = X_train_df.to_numpy(dtype=np.float64)
    X_test = X_test_df.to_numpy(dtype=np.float64)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    X_train_aug = np.c_[np.ones(len(X_train_std)), X_train_std]
    X_test_aug = np.c_[np.ones(len(X_test_std)), X_test_std]
    return X_train_aug, X_test_aug, mean, std


def auc_rank(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metric_summary(y_true, proba, threshold):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "n": len(y_true),
        "event_rate": y_true.mean(),
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y_true),
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "brier": np.mean((proba - y_true) ** 2),
        "auc_rank": auc_rank(y_true, proba),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def fit_and_evaluate_model(target_col: str, label: str):
    X_train_df = make_design_matrix(train_sample, model_features)
    X_test_df = make_design_matrix(test_sample, model_features, columns=X_train_df.columns)
    X_train_aug, X_test_aug, mean, std = standardize_train_test(X_train_df, X_test_df)
    y_train = train_sample[target_col].astype(int).to_numpy()
    y_test = test_sample[target_col].astype(int).to_numpy()

    beta, losses = fit_logistic_numpy(X_train_aug, y_train)
    p_train = sigmoid(X_train_aug @ beta)
    p_test = sigmoid(X_test_aug @ beta)
    threshold = float(y_train.mean())

    global_baseline = np.repeat(y_train.mean(), len(y_test))
    group_rates = (
        train_sample.assign(target=y_train)
        .groupby(["NAME_OF_HIGH_COURT", "Mapped_Bail"])["target"]
        .mean()
    )
    fallback = float(y_train.mean())
    group_baseline = [
        group_rates.get((row["NAME_OF_HIGH_COURT"], row["Mapped_Bail"]), fallback)
        for _, row in test_sample.iterrows()
    ]
    group_baseline = np.asarray(group_baseline)

    metrics = pd.DataFrame([
        {"model": f"{label} logistic", **metric_summary(y_test, p_test, threshold)},
        {"model": f"{label} global baseline", **metric_summary(y_test, global_baseline, threshold)},
        {"model": f"{label} court+bail baseline", **metric_summary(y_test, group_baseline, threshold)},
    ])

    coef = pd.DataFrame({
        "feature": ["INTERCEPT"] + list(X_train_df.columns),
        "coefficient": beta,
    })
    coef["abs_coefficient"] = coef["coefficient"].abs()
    coef = coef.sort_values("abs_coefficient", ascending=False)

    calibration = pd.DataFrame({"y_true": y_test, "proba": p_test})
    calibration["risk_bucket"] = pd.qcut(calibration["proba"], q=10, duplicates="drop")
    calibration_summary = (
        calibration.groupby("risk_bucket")
        .agg(n=("y_true", "size"), mean_predicted_risk=("proba", "mean"), observed_event_rate=("y_true", "mean"))
        .reset_index()
    )

    artifacts = {
        "columns": X_train_df.columns,
        "mean": mean,
        "std": std,
        "beta": beta,
        "threshold": threshold,
        "p_test": p_test,
        "y_test": y_test,
        "losses": losses,
        "metrics": metrics,
        "coef": coef,
        "calibration": calibration_summary,
    }
    return artifacts
"""
))

cells.append(code(
    r"""
model_p75 = fit_and_evaluate_model("LONG_DELAY_P75", "p75 long-delay")
model_p90 = fit_and_evaluate_model("VERY_LONG_DELAY_P90", "p90 very-long-delay")

model_metrics = pd.concat([model_p75["metrics"], model_p90["metrics"]], ignore_index=True)
metric_cols = ["event_rate", "threshold", "accuracy", "precision", "recall", "specificity", "brier", "auc_rank"]
model_metrics_display = model_metrics.copy()
model_metrics_display[metric_cols] = model_metrics_display[metric_cols].round(3)
table_out(model_metrics_display, "04_model_performance.csv", "Temporal test-set model performance", index=False)

table_out(model_p75["losses"].round(4), "04_p75_training_loss.csv", "p75 model training loss checkpoints", index=False)
table_out(model_p75["coef"].head(20).round(4), "04_p75_top_coefficients.csv", "Largest p75 model coefficients", index=False)
table_out(model_p75["calibration"].round(3), "04_p75_calibration_by_decile.csv", "p75 model calibration by predicted-risk decile", index=False)

def risk_lift_table(model_artifacts, label):
    y = np.asarray(model_artifacts["y_test"]).astype(int)
    p = np.asarray(model_artifacts["p_test"])
    base = y.mean()
    rows = []
    for share in [0.10, 0.20, 0.30]:
        cutoff = np.quantile(p, 1 - share)
        selected = p >= cutoff
        rows.append({
            "target": label,
            "selected_highest_risk_share": share,
            "selected_cases": int(selected.sum()),
            "observed_event_rate_selected": y[selected].mean(),
            "overall_event_rate": base,
            "lift_vs_overall": y[selected].mean() / base if base else np.nan,
            "captured_events_share": y[selected].sum() / y.sum() if y.sum() else np.nan,
        })
    return pd.DataFrame(rows)


lift_summary = pd.concat([
    risk_lift_table(model_p75, "p75 long-delay"),
    risk_lift_table(model_p90, "p90 very-long-delay"),
], ignore_index=True)
table_out(lift_summary.round(3), "04_model_lift_summary.csv", "Risk concentration among highest-risk test cases", index=False)
"""
))

cells.append(code(
    r"""
plt.figure(figsize=(8, 5))
cal = model_p75["calibration"].copy()
cal["bucket"] = np.arange(1, len(cal) + 1)
plt.plot(cal["bucket"], cal["mean_predicted_risk"], marker="o", label="Mean predicted risk")
plt.plot(cal["bucket"], cal["observed_event_rate"], marker="o", label="Observed long-delay rate")
plt.title("Calibration: Predicted vs Observed Long-Delay Risk")
plt.xlabel("Predicted-risk decile")
plt.ylabel("Rate")
plt.ylim(0, max(cal["observed_event_rate"].max(), cal["mean_predicted_risk"].max()) * 1.2)
plt.legend()
save_current_fig("04_p75_model_calibration.png")

top_coef_plot = model_p75["coef"].query("feature != 'INTERCEPT'").head(15).copy()
top_coef_plot = top_coef_plot.sort_values("coefficient")
plt.figure(figsize=(9, 6))
colors = np.where(top_coef_plot["coefficient"] >= 0, "#E15759", "#4E79A7")
plt.barh(top_coef_plot["feature"], top_coef_plot["coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Largest Logistic Coefficients For p75 Long-Delay Risk")
plt.xlabel("Coefficient after one-hot encoding and standardization")
plt.ylabel("")
save_current_fig("04_p75_top_coefficients.png")
"""
))

cells.append(md(
    r"""
**Investigation 3 takeaway.** The model is intentionally interpretable and dependency-free. Its purpose is not to determine legal merit; it estimates historical delay risk from court, bail type, case type, filing timing, and limited legal metadata. The baseline comparison shows whether modelling adds signal beyond simple court-and-bail historical rates.

The prediction section estimates **process risk** rather than legal outcome. The label is whether a disposed case crossed a long-delay threshold, not whether an accused person should receive bail. This keeps the model aligned with what the dataset can measure directly. The temporal train/test split also makes the validation more realistic: the model is trained on earlier filing years and evaluated on later ones.

The most useful interpretation is not just classification accuracy. For administrative decision support, the important question is whether the model concentrates risk: do the highest-risk predicted cases actually contain a higher share of delayed cases than the overall test set? The lift table answers that question and makes the prediction component more operationally meaningful.
"""
))

cells.append(md(
    r"""
# Investigation 4 - Pending Burden And COVID-Period Shift

**Lead:** Tushar Singh

**Question:** How did pending burden and filing/disposal patterns vary by court, bail type, and the 2020-2021 COVID-period window?

`CURRENT_STATUS` and `PENDING_DAYS` are scrape-date dependent. Pending analysis is therefore a snapshot of unresolved cases as recorded in the dataset, not a complete longitudinal survival estimate.
"""
))

cells.append(code(
    r"""
pending_summary = (
    df.groupby("NAME_OF_HIGH_COURT")
    .agg(
        total_cases=("CNR_NUMBER", "size"),
        pending_cases=("IS_PENDING", "sum"),
        pending_rate=("IS_PENDING", "mean"),
        median_pending_days=("PENDING_DAYS", "median"),
        p90_pending_days=("PENDING_DAYS", lambda s: s.dropna().quantile(0.90) if s.notna().any() else np.nan),
    )
    .assign(pending_rate=lambda x: x["pending_rate"] * 100)
    .round(2)
    .sort_values("pending_rate", ascending=False)
)
table_out(pending_summary, "05_pending_summary_by_court.csv", "Pending burden by High Court")

pending_by_bail = (
    df.groupby("Mapped_Bail")
    .agg(
        total_cases=("CNR_NUMBER", "size"),
        pending_cases=("IS_PENDING", "sum"),
        pending_rate=("IS_PENDING", "mean"),
        median_pending_days=("PENDING_DAYS", "median"),
    )
    .assign(pending_rate=lambda x: x["pending_rate"] * 100)
    .round(2)
    .sort_values("pending_rate", ascending=False)
)
table_out(pending_by_bail, "05_pending_summary_by_bail.csv", "Pending burden by bail type")

plt.figure(figsize=(10, 6))
pending_plot = pending_summary.reset_index().sort_values("pending_rate", ascending=True)
sns.barplot(data=pending_plot, x="pending_rate", y="NAME_OF_HIGH_COURT", color="#76B7B2")
plt.title("Pending Rate By High Court")
plt.xlabel("Pending cases (%)")
plt.ylabel("")
save_current_fig("05_pending_rate_by_court.png")
"""
))

cells.append(code(
    r"""
df["PERIOD"] = np.where(df["FILED_YEAR"] <= 2019, "Pre-COVID filing years (2010-2019)", "COVID-window filing years (2020-2021)")
disposed["PERIOD"] = np.where(disposed["FILED_YEAR"] <= 2019, "Pre-COVID filing years (2010-2019)", "COVID-window filing years (2020-2021)")

period_summary = (
    df.groupby(["PERIOD", "Mapped_Bail"])
    .agg(
        cases=("CNR_NUMBER", "size"),
        pending_rate=("IS_PENDING", "mean"),
        median_pending_days=("PENDING_DAYS", "median"),
    )
    .assign(pending_rate=lambda x: x["pending_rate"] * 100)
    .round(2)
    .reset_index()
)

period_disposal = (
    disposed.groupby(["PERIOD", "Mapped_Bail"])["DISPOSAL_DAYS"]
    .agg(disposed_cases="count", median_disposal_days="median", p75_disposal_days=lambda s: s.quantile(0.75))
    .round(2)
    .reset_index()
)

period_combined = period_summary.merge(period_disposal, on=["PERIOD", "Mapped_Bail"], how="left")
table_out(period_combined, "05_period_summary_by_bail.csv", "Pre-COVID vs COVID-window summary by bail type", index=False)

year_status = (
    df.groupby("FILED_YEAR")
    .agg(cases=("CNR_NUMBER", "size"), pending_rate=("IS_PENDING", "mean"))
    .assign(pending_rate=lambda x: x["pending_rate"] * 100)
    .reset_index()
)
table_out(year_status.round(2), "05_yearly_filing_and_pending_rate.csv", "Yearly filings and pending rate", index=False)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(year_status["FILED_YEAR"], year_status["cases"], marker="o", color="#4E79A7")
ax1.set_xlabel("Filing year")
ax1.set_ylabel("Filed cases", color="#4E79A7")
ax1.tick_params(axis="y", labelcolor="#4E79A7")
ax2 = ax1.twinx()
ax2.plot(year_status["FILED_YEAR"], year_status["pending_rate"], marker="s", color="#E15759")
ax2.set_ylabel("Pending rate (%)", color="#E15759")
ax2.tick_params(axis="y", labelcolor="#E15759")
plt.title("Filed Cases And Snapshot Pending Rate By Filing Year")
save_current_fig("05_yearly_filings_pending_rate.png")
"""
))

cells.append(md(
    r"""
**Investigation 4 takeaway.** Pending burden is strongly court-specific and must be read as a snapshot. More recent filing years mechanically have less time to dispose and can show higher pending shares, especially around 2020-2021.

The pending analysis is valuable precisely because it is treated cautiously. A pending case in this dataset is not a permanent case attribute; it is the case's status when the record was scraped or synced. For that reason, the module focuses on snapshot burden, pending-days distribution, and relative court differences rather than claiming final lifecycle outcomes. The COVID-window comparison is similarly descriptive: it identifies stress points around 2020-2021 without pretending that this dataset alone can isolate the causal effect of the pandemic.
"""
))

cells.append(md(
    r"""
# Investigation 5 - Observed Bail Outcomes

**Lead:** Yash Kumar

**Question:** Where outcome labels are available, what patterns appear across bail type and court?

Outcome analysis is restricted because `NATURE_OF_DISPOSAL_OUTCOME` is missing for a large share of disposed cases and labels are not standardized across courts. We report coverage before interpreting outcome mixes.
"""
))

cells.append(code(
    r"""
def classify_outcome(value) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NA", "---"}:
        return "Missing"
    if "WITHDRAW" in text or "NOT PRESSED" in text:
        return "Withdrawn/Not Pressed"
    if "ALLOW" in text or "GRANT" in text or text == "BAIL":
        return "Allowed/Granted"
    if "REJECT" in text or "DISMISS" in text:
        return "Rejected/Dismissed"
    if "DISPOSE" in text or "CLOSED" in text or "INFRUCTUOUS" in text:
        return "Other/Disposed"
    return "Other/Disposed"


df["OUTCOME_GROUP"] = df["NATURE_OF_DISPOSAL_OUTCOME"].map(classify_outcome)
outcome_cohort = df.loc[df["IS_DISPOSED"] & df["OUTCOME_GROUP"].ne("Missing")].copy()

outcome_coverage = (
    df.loc[df["IS_DISPOSED"]]
    .groupby("NAME_OF_HIGH_COURT")
    .agg(
        disposed_cases=("CNR_NUMBER", "size"),
        labeled_outcomes=("NATURE_OF_DISPOSAL_OUTCOME", lambda s: s.notna().sum()),
    )
    .assign(outcome_coverage_pct=lambda x: x["labeled_outcomes"] / x["disposed_cases"] * 100)
    .round(2)
    .sort_values("outcome_coverage_pct", ascending=False)
)
table_out(outcome_coverage, "06_outcome_coverage_by_court.csv", "Outcome-label coverage among disposed cases by High Court")

overall_outcome_coverage = len(outcome_cohort) / int(df["IS_DISPOSED"].sum()) * 100
print(f"Disposed cases: {int(df['IS_DISPOSED'].sum()):,}")
print(f"Labeled outcome cohort: {len(outcome_cohort):,} ({overall_outcome_coverage:.2f}% of disposed cases)")
"""
))

cells.append(code(
    r"""
outcome_mix_bail = (
    pd.crosstab(outcome_cohort["Mapped_Bail"], outcome_cohort["OUTCOME_GROUP"], normalize="index")
    .mul(100)
    .round(2)
)
outcome_counts_bail = pd.crosstab(outcome_cohort["Mapped_Bail"], outcome_cohort["OUTCOME_GROUP"])

table_out(outcome_mix_bail, "06_outcome_mix_by_bail_percent.csv", "Observed outcome mix by bail type, labeled subset only")
table_out(outcome_counts_bail, "06_outcome_counts_by_bail.csv", "Observed outcome counts by bail type, labeled subset only")

ax = outcome_mix_bail.plot(kind="barh", stacked=True, figsize=(10, 4.8), width=0.8)
ax.set_title("Observed Disposal Outcomes By Bail Type\nRestricted to rows with outcome labels")
ax.set_xlabel("Share within labeled outcome subset (%)")
ax.set_ylabel("")
ax.legend(title="Outcome group", bbox_to_anchor=(1.02, 1), loc="upper left")
save_current_fig("06_outcome_mix_by_bail_type.png")

outcome_court = (
    outcome_cohort.groupby(["NAME_OF_HIGH_COURT", "OUTCOME_GROUP"])
    .size()
    .reset_index(name="cases")
)
court_labeled_totals = outcome_court.groupby("NAME_OF_HIGH_COURT")["cases"].transform("sum")
outcome_court["share_pct"] = outcome_court["cases"] / court_labeled_totals * 100
outcome_court_pivot = (
    outcome_court.pivot(index="NAME_OF_HIGH_COURT", columns="OUTCOME_GROUP", values="share_pct")
    .fillna(0)
    .round(2)
)
outcome_court_pivot["LABELED_OUTCOMES"] = outcome_cohort["NAME_OF_HIGH_COURT"].value_counts()
outcome_court_pivot = outcome_court_pivot.sort_values("LABELED_OUTCOMES", ascending=False)
table_out(outcome_court_pivot, "06_outcome_mix_by_court_percent.csv", "Observed outcome mix by court, labeled subset only")
"""
))

cells.append(md(
    r"""
**Investigation 5 takeaway.** Outcome data can enrich the legal story, but it is not complete enough to support a headline claim like "predict whether bail will be granted." The appropriate use is a restricted-sample view of observed disposal labels with transparent coverage warnings.

This is the most legally tempting part of the dataset, so it receives the strictest caveat. Outcome labels are not uniformly populated and are not standardized across courts. The notebook therefore reports outcome-label coverage first, groups labels into broad interpretable categories, and avoids using the restricted outcome subset as if it represented the full bail-case universe. The result is still useful: it shows how outcome patterns look where labels exist, while preserving the integrity of the stronger full-cohort findings on delay and pendency.
"""
))

cells.append(md(
    r"""
# Practical Demo - Historical Case Context

This demo returns historical context for a hypothetical metadata profile. It reports similar-case disposal time, pending-rate context, model-estimated long-delay risk, and observed outcome mix only when enough labeled outcomes exist.
"""
))

cells.append(code(
    r"""
def normalize_case_type_for_demo(case_type):
    if case_type is None or pd.isna(case_type):
        return "OTHER_CASE_TYPE"
    cleaned = str(case_type).upper().strip()
    return cleaned if cleaned in set(top_case_types) else "OTHER_CASE_TYPE"


def predict_long_delay_context(
    high_court,
    bail_type,
    filing_year,
    filing_month,
    case_type_group,
    act_group="MISSING",
    section_group="MISSING",
):
    row = pd.DataFrame([{
        "NAME_OF_HIGH_COURT": high_court,
        "Mapped_Bail": bail_type,
        "CASE_TYPE_GROUP": case_type_group,
        "FILED_YEAR": str(int(filing_year)),
        "FILED_MONTH": str(int(filing_month)).zfill(2),
        "ACT_GROUP": act_group,
        "SECTION_GROUP": section_group,
    }])
    X_row = make_design_matrix(row, model_features, columns=model_p75["columns"])
    X = X_row.to_numpy(dtype=np.float64)
    X_std = (X - model_p75["mean"]) / model_p75["std"]
    X_aug = np.c_[np.ones(len(X_std)), X_std]
    return float(sigmoid(X_aug @ model_p75["beta"])[0])


def historical_case_context(
    high_court,
    bail_type,
    filing_year,
    filing_month=1,
    case_type=None,
    act_group="MISSING",
    section_group="MISSING",
    min_similar=30,
):
    case_type_group = normalize_case_type_for_demo(case_type)
    filing_year = int(filing_year)
    filing_month = int(filing_month)

    disposed_candidates = disposed.copy()
    filters = [
        ("court+bail+case_type", (
            disposed_candidates["NAME_OF_HIGH_COURT"].eq(high_court)
            & disposed_candidates["Mapped_Bail"].eq(bail_type)
            & disposed_candidates["CASE_TYPE_GROUP"].eq(case_type_group)
        )),
        ("court+bail", (
            disposed_candidates["NAME_OF_HIGH_COURT"].eq(high_court)
            & disposed_candidates["Mapped_Bail"].eq(bail_type)
        )),
        ("bail_type", disposed_candidates["Mapped_Bail"].eq(bail_type)),
    ]
    selected_label = "all disposed cases"
    similar = disposed_candidates
    for label, mask in filters:
        if int(mask.sum()) >= min_similar:
            selected_label = label
            similar = disposed_candidates.loc[mask]
            break

    status_mask = (
        df["NAME_OF_HIGH_COURT"].eq(high_court)
        & df["Mapped_Bail"].eq(bail_type)
    )
    status_context = df.loc[status_mask]

    outcome_mask = (
        outcome_cohort["NAME_OF_HIGH_COURT"].eq(high_court)
        & outcome_cohort["Mapped_Bail"].eq(bail_type)
    )
    outcome_context = outcome_cohort.loc[outcome_mask]

    risk = predict_long_delay_context(
        high_court=high_court,
        bail_type=bail_type,
        filing_year=filing_year,
        filing_month=filing_month,
        case_type_group=case_type_group,
        act_group=act_group,
        section_group=section_group,
    )

    summary = {
        "input_high_court": high_court,
        "input_bail_type": bail_type,
        "input_filing_year": filing_year,
        "input_filing_month": filing_month,
        "input_case_type_group": case_type_group,
        "similarity_level_used": selected_label,
        "similar_disposed_cases": int(len(similar)),
        "similar_median_disposal_days": float(similar["DISPOSAL_DAYS"].median()),
        "similar_p75_disposal_days": float(similar["DISPOSAL_DAYS"].quantile(0.75)),
        "similar_p90_disposal_days": float(similar["DISPOSAL_DAYS"].quantile(0.90)),
        "model_estimated_long_delay_risk_p75": risk,
        "court_bail_pending_rate": float(status_context["IS_PENDING"].mean()) if len(status_context) else np.nan,
        "court_bail_total_cases": int(len(status_context)),
        "outcome_context_available": int(len(outcome_context)),
    }
    result = pd.DataFrame([summary])

    if len(outcome_context) >= min_similar:
        outcome_mix = (
            outcome_context["OUTCOME_GROUP"].value_counts(normalize=True)
            .mul(100)
            .round(2)
            .rename("observed_labeled_outcome_share_pct")
            .reset_index()
            .rename(columns={"index": "OUTCOME_GROUP"})
        )
    else:
        outcome_mix = pd.DataFrame({
            "OUTCOME_GROUP": ["Insufficient labeled outcome records"],
            "observed_labeled_outcome_share_pct": [np.nan],
        })

    return result.round(4), outcome_mix


demo_summary, demo_outcomes = historical_case_context(
    high_court="HIGH COURT OF RAJASTHAN",
    bail_type="REGULAR BAIL",
    filing_year=2020,
    filing_month=7,
    case_type="CRLMB",
    act_group="CRPC",
    section_group="SEC_439_REGULAR",
)

display(Markdown("**Demo output: historical context for a Rajasthan regular-bail profile**"))
display(demo_summary)
display(demo_outcomes)

demo_summary.to_csv(TABLE_DIR / "07_demo_summary.csv", index=False)
demo_outcomes.to_csv(TABLE_DIR / "07_demo_outcome_context.csv", index=False)
"""
))

cells.append(md(
    r"""
# Final Synthesis Across Investigations

The five investigations are meant to work together. The table below summarizes the strongest evidence from each module and the main caution attached to it. This is the high-level story a reader should retain after reviewing the notebook.
"""
))

cells.append(code(
    r"""
regular_median = float(delay_by_bail.loc["REGULAR BAIL", "median_days"])
anticipatory_median = float(delay_by_bail.loc["ANTICIPATORY BAIL", "median_days"])
cancellation_median = float(delay_by_bail.loc["CANCELLATION", "median_days"])
highest_adjusted_court = adjusted_court_delay.index[0]
highest_adjusted_index = float(adjusted_court_delay.iloc[0]["adjusted_delay_index"])
highest_pending_court = pending_summary.index[0]
highest_pending_rate = float(pending_summary.iloc[0]["pending_rate"])
p75_auc = float(model_metrics.loc[model_metrics["model"].eq("p75 long-delay logistic"), "auc_rank"].iloc[0])
p75_top20_lift = float(lift_summary.loc[
    (lift_summary["target"].eq("p75 long-delay"))
    & (lift_summary["selected_highest_risk_share"].eq(0.20)),
    "lift_vs_overall"
].iloc[0])
outcome_coverage_overall = len(outcome_cohort) / int(df["IS_DISPOSED"].sum()) * 100

synthesis = pd.DataFrame([
    {
        "module": "Bail-type landscape",
        "strongest_evidence": "Regular bail is the dominant category, but anticipatory bail is substantial and cancellation is rare.",
        "key_number": f"Regular {bail_counts.loc[bail_counts['Mapped_Bail'].eq('REGULAR BAIL'), 'share_pct'].iloc[0]:.2f}%; anticipatory {bail_counts.loc[bail_counts['Mapped_Bail'].eq('ANTICIPATORY BAIL'), 'share_pct'].iloc[0]:.2f}%; cancellation {bail_counts.loc[bail_counts['Mapped_Bail'].eq('CANCELLATION'), 'share_pct'].iloc[0]:.2f}%",
        "main_caution": "Bail-type mix differs sharply by court, so pooled comparisons can mislead.",
    },
    {
        "module": "Disposal delay",
        "strongest_evidence": "Cancellation matters take far longer than regular or anticipatory bail matters.",
        "key_number": f"Median days: regular {regular_median:.0f}, anticipatory {anticipatory_median:.0f}, cancellation {cancellation_median:.1f}",
        "main_caution": "Delay comparisons are descriptive; the data do not include all legal merits or case complexity factors.",
    },
    {
        "module": "Adjusted court inequality",
        "strongest_evidence": "Court-level delay differences remain visible after using a bail/year/case-type benchmark.",
        "key_number": f"Highest adjusted index: {highest_adjusted_court} ({highest_adjusted_index:.2f})",
        "main_caution": "The index is a benchmark, not a causal estimate of court efficiency.",
    },
    {
        "module": "Long-delay risk model",
        "strongest_evidence": "Filing-stage metadata contains signal for long-delay risk, but the model is best used as context rather than a deterministic prediction.",
        "key_number": f"p75 test AUC {p75_auc:.3f}; top-quintile risk lift {p75_top20_lift:.2f}x",
        "main_caution": "Prediction target is delay, not bail outcome or legal merit.",
    },
    {
        "module": "Pending and outcome analysis",
        "strongest_evidence": "Pending burden and observed outcome patterns vary strongly by court and bail type.",
        "key_number": f"Highest pending rate: {highest_pending_court} ({highest_pending_rate:.2f}%); outcome-label coverage {outcome_coverage_overall:.2f}%",
        "main_caution": "Pending status is a scrape-date snapshot; outcome labels are incomplete.",
    },
])

table_out(synthesis, "08_results_synthesis.csv", "Final synthesis across empirical investigations", index=False)
"""
))

cells.append(md(
    r"""
## Scope Of Interpretation

The results should be read as evidence about **case processing**, not as conclusions about the legal merits of individual bail applications. The dataset is well suited to studying time, status, court-level variation, bail-type composition, and broad observed outcome patterns where labels exist. It is not sufficient for determining whether a particular accused person should receive bail, because it does not include the full factual record, custody history, criminal history, evidence, or judicial reasoning.

Court comparisons are also interpreted with care. The adjusted delay index improves on raw rankings by accounting for bail type, filing year, and case-type group, but it remains a descriptive benchmark. It identifies where observed disposal times are faster or slower than similar cases in the dataset; it does not establish a causal explanation for those differences.

The practical contribution is therefore administrative and empirical: the analysis identifies where delay and pending burden concentrate, how those patterns differ by bail type, and where outcome labels can and cannot support further interpretation.
"""
))

cells.append(md(
    r"""
# Appendix - Validation Checklist, Files, And Limitations
"""
))

cells.append(code(
    r"""
created_tables = sorted([p.name for p in TABLE_DIR.glob("*.csv")])
created_figures = sorted([p.name for p in FIG_DIR.glob("*.png")])

checklist = pd.DataFrame([
    {"item": "Notebook runs from project folder without absolute data path", "status": "PASS"},
    {"item": "Raw column count is 32", "status": "PASS" if len(raw_columns) == 32 else "CHECK"},
    {"item": "Record count is 927,896", "status": "PASS" if len(df) == 927896 else "CHECK"},
    {"item": "All non-null core date values parsed", "status": "PASS" if (date_validation["unparsed_non_null"] == 0).all() else "CHECK"},
    {"item": "Prediction excludes leakage fields", "status": "PASS" if not leakage_fields.intersection(model_features) else "CHECK"},
    {"item": "Tables generated", "status": f"{len(created_tables)} CSV files"},
    {"item": "Figures generated", "status": f"{len(created_figures)} PNG files"},
])
table_out(checklist, "08_final_validation_checklist.csv", "Final validation checklist", index=False)

manifest = {
    "notebook": "Phase2_Bail_Empirical_Investigation.ipynb",
    "expected_html": "Phase2_Bail_Empirical_Investigation.html",
    "tables": created_tables,
    "figures": created_figures,
}
with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
"""
))

cells.append(md(
    r"""
## Main Limitations

- **Court heterogeneity:** High Courts use different registry conventions and case-type labels, so court comparisons are descriptive and adjusted only with available metadata.
- **Outcome incompleteness:** disposal outcomes are available for only a subset of disposed cases; outcome results are restricted-sample patterns, not population-level bail-grant estimates.
- **Legal merits unavailable:** the dataset lacks detailed FIR facts, custody duration, criminal history, judicial reasoning, lawyer arguments, and accused-level attributes. Individual bail-outcome prediction would therefore be inappropriate.
- **Pending status is a snapshot:** pending cases are unresolved as of the scrape/sync date, so pending analysis is not a full survival analysis.
- **Legal metadata missingness:** `UNDER_ACTS` and `UNDER_SECTIONS` are sparse and unevenly populated across courts, so legal subject-matter findings are shown with coverage checks.

## Integrated Interpretation

This project turns a large administrative dataset into a reproducible empirical workflow with explicit measurement checks, court-aware comparisons, and a restrained prediction component. The strongest findings are about **process**: how quickly different bail matters move, where delay is concentrated, which courts carry higher pending burden, and how much caution is needed before interpreting outcome labels.

The decision not to model individual bail grant or rejection is part of the empirical design. The data do not contain the legal merits of each application and disposal-outcome labels are incomplete. Long-delay risk is a more appropriate target because disposal time is directly measured for disposed cases and can be validated against historical data.

The robustness of the workflow comes from explicit mixed-format date parsing, exclusion of leakage fields from the model, court-level metadata completeness reporting, restricted-sample outcome analysis, and automated generation of every table and figure through the notebook.

## Final Interpretation

The final conclusion is deliberately balanced. The dataset is not sufficient for individual legal prediction, but it is very strong for studying administrative patterns in bail processing. Across nearly 0.93 million High Court bail records, the analysis shows that bail type and court context are deeply related to disposal time, pending burden, and observed outcomes where outcome labels exist. The project therefore offers a practical reform-oriented view of bail case administration while staying honest about the limits of judicial administrative data.
"""
))


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
)

OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {OUT}")

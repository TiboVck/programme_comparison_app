import io
from datetime import datetime
import textwrap
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import numpy as np
import plotly.express as px
from  PIL import Image


# =========================
# CONFIG
# =========================
B1_name = "D&C Dep ID - Pre to O&M"
B2_name = "Input req from D&C"
B3_name = "D&C Dep ID - Full List"
B4_name = "D&C Dep ID - Key Milestone"

ID_COL = "Activity ID"  # <-- change if needed
DATE_COLS = ["Start", "Finish"]  # <-- change if needed

# Which columns to compare (only those present will be used)
COLS_TO_COMPARE = [
    "Start",
    "Finish",
    "Activity % Complete"

]

# Keep only rows where ID starts with a 5 (as you asked earlier)
FILTER_ID_STARTS_WITH = "5"  # set to None to disable

ROW_HEIGHT = 22          # hauteur par activité (ajuste si besoin)
MAX_VISIBLE = 40         # nb de lignes visibles à l’écran avant scroll
ROW_HEIGHT = 22   # hauteur par activité (ajuste si tu veux plus serré)
WINDOW_ROWS = 40  # nb de lignes visibles avant scroll

# =========================
# DATA CLEANING
# =========================

def remove_fully_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are 100% empty (all columns empty)."""
    df = df.copy()
    df = df.dropna(how="all")
    return df


def clean_df(df: pd.DataFrame, id_col: str = ID_COL) -> pd.DataFrame:
    """Clean dataframe for safe merging and comparisons."""
    df = df.copy()

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Remove fully empty rows
    df = remove_fully_empty_rows(df)

    # Normalize all object/string cells (trim)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")
            df[col] = (
                df[col]
                .str.replace("\u00A0", " ", regex=False)
                .str.replace("\u200B", "", regex=False)
                .str.strip()
            )

    # Normalize ID
    if id_col in df.columns:
        df[id_col] = df[id_col].astype("string")
        df[id_col] = (
            df[id_col]
            .str.replace("\u00A0", " ", regex=False)
            .str.replace("\u200B", "", regex=False)
            .str.strip()
        )

    # Convert dates
    for date_col in DATE_COLS:
        if date_col in df.columns:

            # 1. Nettoyage uniquement des strings
            mask_str = df[date_col].apply(lambda x: isinstance(x, str))

            df.loc[mask_str, date_col] = (
                df.loc[mask_str, date_col]
                .str.strip()
                .str.replace(r"\s*A\s*$", "", regex=True)  # enlève " A"
                .str.replace(r"\*", "", regex=True)
                .str.replace("\u00A0", " ", regex=False)
            )

            # 2. Premier parsing (format attendu type 03-Nov-25)
            parsed = pd.to_datetime(
                df[date_col],
                errors="coerce",
                format="%d-%b-%y"
            )

            # 3. Fallback pour ce qui a échoué
            fallback = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

            # 4. Combine les deux
            df[date_col] = parsed.fillna(fallback)

    # Debug des erreurs de parsing
    for col in DATE_COLS:
        if col in df.columns:
            failed = df[df[col].isna()]
            if not failed.empty:
                print(f"\n❌ Failed parsing in {col}:")
                print(failed[[col]].head(10))

    return df


# =========================
# COMPARISON
# =========================

def compare_versions_simple(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
):

    # --- FILTER ---
    df_old = df_old[df_old[ID_COL].notna()].copy()
    df_new = df_new[df_new[ID_COL].notna()].copy()

    if FILTER_ID_STARTS_WITH:
        df_old = df_old[df_old[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()
        df_new = df_new[df_new[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()

    # --- KEEP COLUMNS ---
    old_keep = [ID_COL, "Activity Name"] + [c for c in COLS_TO_COMPARE if c in df_old.columns]
    new_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_new.columns]

    df_old = df_old[old_keep].copy()
    df_new = df_new[new_keep].copy()

    # Defensive: ensure ID is string
    df_old[ID_COL] = df_old[ID_COL].astype("string").str.strip()
    df_new[ID_COL] = df_new[ID_COL].astype("string").str.strip()

    # --- MERGE ---
    merged = df_old.merge(
        df_new,
        on=ID_COL,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )

    # --- STATUS ---
    merged["Status"] = merged["_merge"].map({
        "left_only": "REMOVED",
        "right_only": "ADDED",
        "both": "EXISTING",
    }).astype(str)

    # ============================================================
    # 🔥 COMPARISON
    # ============================================================

    for col in COLS_TO_COMPARE:
        old_col = f"{col}_old"
        new_col = f"{col}_new"
        change_col = f"{col}_change"
        diff_col = f"{col}_diff"

        merged[change_col] = "No Change"
        merged[diff_col] = pd.NA

        # --- DATES ---
        if col in ["Start", "Finish"]:
            old_date = pd.to_datetime(merged[old_col], errors="coerce")
            new_date = pd.to_datetime(merged[new_col], errors="coerce")

            changed = ~(
                (old_date.dt.date == new_date.dt.date) |
                (old_date.isna() & new_date.isna())
            )

            merged[change_col] = np.where(changed, "Change", "No Change")

            diff_days = (new_date - old_date).dt.days
            merged[diff_col] = diff_days.astype("Int64")

        # --- % COMPLETE ---
        elif col == "Activity % Complete":
            old_num = pd.to_numeric(merged[old_col], errors="coerce")
            new_num = pd.to_numeric(merged[new_col], errors="coerce")

            changed = ~(
                (old_num == new_num) |
                (old_num.isna() & new_num.isna())
            )

            merged[change_col] = np.where(changed, "Change", "No Change")

            diff_pts = new_num - old_num
            merged[diff_col] = diff_pts.apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else x
            )

    # ============================================================
    # --- GLOBAL CHANGE ---
    change_cols = [f"{c}_change" for c in COLS_TO_COMPARE]

    merged["Any_change"] = merged[change_cols].apply(
        lambda row: any(v == "Change" for v in row), axis=1
    )

    merged.loc[(merged["Status"] == "EXISTING") & (merged["Any_change"]), "Status"] = "MODIFIED"
    merged.loc[(merged["Status"] == "EXISTING") & (~merged["Any_change"]), "Status"] = "UNCHANGED"

    # --- SORT ---
    order = {"MODIFIED": 0, "ADDED": 1, "REMOVED": 2, "UNCHANGED": 3}
    merged["sort"] = merged["Status"].map(order)

    merged = merged.sort_values(["sort", ID_COL]).drop(columns=["sort"])

    return merged


def compare_versions_very_focused(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    bucket_sheets: dict   # <-- NEW INPUT
):

    # --- FILTER ---
    df_old = df_old[df_old[ID_COL].notna()].copy()
    df_new = df_new[df_new[ID_COL].notna()].copy()

    if FILTER_ID_STARTS_WITH:
        df_old = df_old[df_old[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()
        df_new = df_new[df_new[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()

        # Apply same filter to ALL bucket sheets
        for name, df in bucket_sheets.items():
            bucket_sheets[name] = df[
                df[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)
            ].copy()

    # --- KEEP COLUMNS ---
    old_keep = [ID_COL, "Activity Name"] + [c for c in COLS_TO_COMPARE if c in df_old.columns]
    new_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_new.columns]

    df_old = df_old[old_keep].copy()
    df_new = df_new[new_keep].copy()

    # --- BUILD BUCKET SETS DYNAMICALLY ---
    bucket_sets = {}
    for name, df in bucket_sheets.items():
        bucket_sets[name] = set(df[ID_COL].dropna().astype(str).str.strip())

    # Defensive: ensure ID is string
    df_old[ID_COL] = df_old[ID_COL].astype("string").str.strip()
    df_new[ID_COL] = df_new[ID_COL].astype("string").str.strip()

    # --- MERGE ---
    merged = df_old.merge(
        df_new,
        on=ID_COL,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )

    # --- STATUS ---
    merged["Status"] = merged["_merge"].map({
        "left_only": "REMOVED",
        "right_only": "ADDED",
        "both": "EXISTING",
    }).astype(str)

    # --- ASSIGN BUCKET (DYNAMIC) ---
    def get_bucket(activity_id):
        activity_id = str(activity_id).strip()
        for bucket_name, id_set in bucket_sets.items():
            if activity_id in id_set:
                return bucket_name
        return "OTHER"

    merged["bucket"] = merged[ID_COL].apply(get_bucket)

    # ============================================================
    # 🔥 COMPARISON
    # ============================================================

    for col in COLS_TO_COMPARE:
        old_col = f"{col}_old"
        new_col = f"{col}_new"
        change_col = f"{col}_change"
        diff_col = f"{col}_diff"

        merged[change_col] = "No Change"
        merged[diff_col] = pd.NA

        if col in ["Start", "Finish"]:
            old_date = pd.to_datetime(merged[old_col], errors="coerce")
            new_date = pd.to_datetime(merged[new_col], errors="coerce")

            changed = ~(
                (old_date.dt.date == new_date.dt.date) |
                (old_date.isna() & new_date.isna())
            )

            merged[change_col] = np.where(changed, "Change", "No Change")

            diff_days = (new_date - old_date).dt.days
            merged[diff_col] = diff_days.astype("Int64")

        elif col == "Activity % Complete":
            old_num = pd.to_numeric(merged[old_col], errors="coerce")
            new_num = pd.to_numeric(merged[new_col], errors="coerce")

            changed = ~(
                (old_num == new_num) |
                (old_num.isna() & new_num.isna())
            )

            merged[change_col] = np.where(changed, "Change", "No Change")

            diff_pts = new_num - old_num
            merged[diff_col] = diff_pts.apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else x
            )

    # ============================================================
    # --- GLOBAL CHANGE ---
    change_cols = [f"{c}_change" for c in COLS_TO_COMPARE]
    merged["Any_change"] = merged[change_cols].apply(
        lambda row: any(v == "Change" for v in row), axis=1
    )
    
    merged.loc[(merged["Status"] == "EXISTING") & (merged["Any_change"]), "Status"] = "MODIFIED"
    merged.loc[(merged["Status"] == "EXISTING") & (~merged["Any_change"]), "Status"] = "UNCHANGED"

    # --- SORT ---
    order = {"MODIFIED": 0, "ADDED": 1, "REMOVED": 2, "UNCHANGED": 3}
    merged["sort"] = merged["Status"].map(order)
    merged = merged.sort_values(["sort", ID_COL]).drop(columns=["sort"])

    # --- SPLIT DYNAMICALLY ---
    bucket_results = {
        name: merged[merged["bucket"] == name].copy()
        for name in bucket_sets.keys()
    }

    other_changes = merged[merged["bucket"] == "OTHER"].copy()

    return merged, bucket_results, other_changes


import pandas as pd

def duplicate_new_old_rows(df):
    # --- NEW ---
    df_new = pd.DataFrame({
        "Task": df["Activity Name_new"].astype(str) + " NEW",
        "Task Description": "",
        "Start": df["Start_new"],
        "Finish": df["Finish_new"],
        "Completion Pct": "",
        "Team": ""
    })

    # --- OLD ---
    df_old = pd.DataFrame({
        "Task": df["Activity Name_old"].astype(str),
        "Task Description": "",
        "Start": df["Start_old"],
        "Finish": df["Finish_old"],
        "Completion Pct": "",
        "Team": ""
    })

    df_new["Version"] = "NEW"
    df_old["Version"] = "OLD"


    # Concat NEW puis OLD ligne par ligne
    out = pd.concat([df_new, df_old], ignore_index=True)

    out = out.iloc[
        [i for pair in zip(range(len(df_new)), range(len(df_new), len(out))) for i in pair]
    ].reset_index(drop=True)

    return out


def to_excel_bytes(summary_df: pd.DataFrame, prio_changes: pd.DataFrame, other_changes: pd.DataFrame) -> bytes:
    """Return an Excel file as bytes (for Streamlit download)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)
        prio_changes.to_excel(writer, sheet_name="Prioritised", index=False)
        other_changes.to_excel(writer, sheet_name="Other", index=False)
    buffer.seek(0)
    return buffer.read()

def to_excel_bytes_vf(
    df_all: pd.DataFrame,
    bucket_results: dict,   # <-- NEW (dict of {sheet_name: df})
    df_other: pd.DataFrame,
) -> bytes:
    """Return an Excel file as bytes (for Streamlit download)."""

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Main sheet
        df_all.to_excel(writer, sheet_name="ALL", index=False)

        # Write each bucket dynamically
        for sheet_name, df in bucket_results.items():
            # Excel sheet name limit = 31 chars
            safe_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

        # OTHER sheet
        df_other.to_excel(writer, sheet_name="OTHER", index=False)

    buffer.seek(0)
    return buffer.read()

def to_excel_bytes_vf_simple(
    df_summary: pd.DataFrame,   
    df_comparison: pd.DataFrame,
) -> bytes:
    """Return an Excel file as bytes (for Streamlit download)."""

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Main sheet
        df_summary.to_excel(writer, sheet_name="SUMMARY", index=False)

        df_comparison.to_excel(writer, sheet_name="COMPARISON", index=False)

    buffer.seek(0)
    return buffer.read()



# =========================
# STREAMLIT UI
# =========================

import streamlit as st

st.set_page_config(
    page_title="My App",
    layout="wide",
    initial_sidebar_state="collapsed"  # 👈 sidebar cachée par défaut
)

st.title("📊 Excel Program Comparator")
st.caption("Upload OLD and NEW versions of your program, click Compare, get a clean Excel diff + charts.")

with st.sidebar:
    st.header("Settings")
    st.write("Adjust these if your file uses different column names.")

    ID_COL = st.text_input("ID column name", value=ID_COL)

    cols_default = ",".join(COLS_TO_COMPARE)
    cols_raw = st.text_area("Columns to compare (comma-separated)", value=cols_default)
    COLS_TO_COMPARE = [c.strip() for c in cols_raw.split(",") if c.strip()]

    starts_with = st.text_input("Keep only IDs starting with (optional)", value=FILTER_ID_STARTS_WITH or "")
    FILTER_ID_STARTS_WITH = starts_with.strip() or None

    remove_unchanged = st.checkbox("Remove UNCHANGED rows", value=True)


col1, col2, col3 = st.columns(3)

with col1:
    old_file = st.file_uploader("⬆️ Upload OLD version", type=["xlsx"], key="old")

with col2:
    new_file = st.file_uploader("⬆️ Upload NEW version", type=["xlsx"], key="new")

with col3:
    focused_ID = st.file_uploader("⬆️ Upload focused ID List", type=["xlsx"], key="focused")

compare_clicked = st.button("🚀 Compare", type="primary", use_container_width=True)

if compare_clicked:
    if not old_file or not new_file :
        st.error("Please upload BOTH files before comparing.")
        st.stop()

    #Version non focused
    if not focused_ID:

        try:
            df_old = pd.read_excel(old_file)
            df_new = pd.read_excel(new_file)

            df_old = clean_df(df_old, id_col=ID_COL)
            df_new = clean_df(df_new, id_col=ID_COL)

            # Guard: ID column must exist
            if ID_COL not in df_old.columns or ID_COL not in df_new.columns:
                st.error(f"Column '{ID_COL}' must exist in BOTH files.")
                st.stop()

            # Compare
            df_comparison = compare_versions_simple(df_old, df_new)

            # Optionally remove unchanged in UI (already removed by default in compare_versions)
            if not remove_unchanged:
                pass

            # Summary
            df_summary = (
                df_comparison["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
            )

            # KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("Modified", int((df_comparison["Status"] == "MODIFIED").sum()))
            k2.metric("Added", int((df_comparison["Status"] == "ADDED").sum()))
            k3.metric("Removed", int((df_comparison["Status"] == "REMOVED").sum()))

            # Charts
            st.subheader("📈 Changes breakdown")

            # Tri (optionnel) par nombre décroissant pour une lecture plus claire
            df_plot = df_summary.sort_values("Count", ascending=False)

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(df_plot["Status"], df_plot["Count"], color="#4C78A8")

            # Ajoute les valeurs (ou %) au-dessus des barres
            total = df_plot["Count"].sum()
            for bar, val in zip(bars, df_plot["Count"]):
                pct = val / total * 100 if total else 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

            ax.set_xlabel("Status")
            ax.set_ylabel("Count")
            ax.set_title("Status Distribution")
            ax.set_ylim(0, max(df_plot["Count"]) * 1.15)  # un peu d'espace pour les labels
            plt.xticks(rotation=20)

            st.pyplot(fig)

            # Table
            st.subheader("🧾 Detailed comparison")
            st.dataframe(df_comparison, use_container_width=True, height=520)

            # Download
            excel_bytes = to_excel_bytes_vf_simple(df_summary, df_comparison)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"program_comparison_{ts}.xlsx"

            st.download_button(
                label="⬇️ Download Excel result",
                data=excel_bytes,
                file_name=out_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            st.success("Done!")

        except Exception as e:
            st.exception(e)

    #Version FOCUSED
    else :

        try:
            df_old = pd.read_excel(old_file)
            df_new = pd.read_excel(new_file)

            xls = pd.ExcelFile(focused_ID)

            buckets_sheets = {}

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                buckets_sheets[sheet_name] = clean_df(df, id_col=ID_COL)


            df_old_clean = clean_df(df_old, id_col=ID_COL)
            df_new_clean = clean_df(df_new, id_col=ID_COL)
            

            # Guard: ID column must exist
            if ID_COL not in df_old.columns or ID_COL not in df_new.columns:
                st.error(f"Column '{ID_COL}' must exist in BOTH files.")
                st.stop()

            # Compare
            merged, bucket_results, other_changes = compare_versions_very_focused(
                df_old,
                df_new,
                buckets_sheets
            )

            df_summary = (
                merged["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
            )

            # KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("Modified", int((merged["Status"] == "MODIFIED").sum()))
            k2.metric("Added", int((merged["Status"] == "ADDED").sum()))
            k3.metric("Removed", int((merged["Status"] == "REMOVED").sum()))

            # Charts
            st.subheader("📈 Changes breakdown")
            # Tri (optionnel) par nombre décroissant pour une lecture plus claire
            df_plot = df_summary.sort_values("Count", ascending=False)

            fig, ax = plt.subplots(figsize=(3, 2))
            bars = ax.bar(df_plot["Status"], df_plot["Count"], color="#4C78A8")

            # Ajoute les valeurs (ou %) au-dessus des barres
            total = df_plot["Count"].sum()
            for bar, val in zip(bars, df_plot["Count"]):
                pct = val / total * 100 if total else 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

            ax.set_xlabel("Status")
            ax.set_ylabel("Count")
            ax.set_title("Status Distribution")
            ax.set_ylim(0, max(df_plot["Count"]) * 1.15)  # un peu d'espace pour les labels
            plt.xticks(rotation=20)

            st.pyplot(fig, use_container_width=False)

            # Table
            st.subheader("🧾 Focused comparison")

            if bucket_results:
                first_bucket_name = next(iter(bucket_results))
                first_bucket_df = bucket_results[first_bucket_name]

                st.markdown(f"### {first_bucket_name}")
                st.dataframe(first_bucket_df, use_container_width=True, height=520)

            # Download
            excel_bytes = to_excel_bytes_vf(
                df_all=merged,
                bucket_results=bucket_results,
                df_other=other_changes
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"program_comparison_{ts}.xlsx"

            st.download_button(
                label="⬇️ Download Excel result",
                data=excel_bytes,
                file_name=out_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            st.success("Done!")

            Tasks = duplicate_new_old_rows(first_bucket_df.copy())

            Tasks['Start'] = pd.to_datetime(Tasks['Start'])
            Tasks['Finish'] = pd.to_datetime(Tasks['Finish'])

            with st.expander("📋 View generated tasks", expanded=False):
                st.dataframe(
                    Tasks,
                    use_container_width=True,
                    height=400
                )
            Tasks=duplicate_new_old_rows(first_bucket_df.copy())

            Tasks['Start'] = Tasks['Start'].astype('datetime64[ns]')
            Tasks['Finish'] = Tasks['Finish'].astype('datetime64[ns]')
            
            # grid_response = AgGrid(
            #    Tasks,
            #   editable=True, 
            #    height=300, 
            #    width='100%',
            #    )

            #updated = grid_response['data'] """
            df = Tasks.copy()
            
            #Main interface - section 3
            st.subheader('Step 3: Generate the Gantt chart')
            
            
            # hauteur dynamique (si tu veux tout voir en 1 écran -> mets juste height = len(df)*ROW_HEIGHT)
            height = 500
            WRAP_WIDTH = 40

            # wrap texte + alignement gauche
            df["Task_wrapped"] = df["Task"].apply(
                lambda x: "<br>".join([x[i:i+WRAP_WIDTH] for i in range(0, len(x), WRAP_WIDTH)])
            )

            y_order = df["Task_wrapped"].tolist()

            fig = px.timeline(
                df,
                x_start="Start",
                x_end="Finish",
                y="Task_wrapped",
                color="Version",
                hover_name="Task",
                category_orders={"Task_wrapped": y_order},
                color_discrete_map={
                    "NEW": "red",
                    "OLD": "blue"
                }
            )

            fig.update_yaxes(autorange="reversed")

            # zoom initial -> 10 lignes visibles
            fig.update_yaxes(
                range=[10, -1],
                fixedrange=False
            )

            fig.update_layout(
                title="Project Plan Gantt Chart",
                height=height,
                bargap=0.2,
                title_x=0.5,
                dragmode="pan",

                # police globale
                font=dict(
                    family="Arial",
                    size=13
                ),

                xaxis=dict(
                    rangeslider_visible=True,
                    side="top",
                    showgrid=True,
                    zeroline=True,
                    showline=True,
                    tickformat="%x\n"
                ),

                # labels à gauche
                yaxis=dict(
                    automargin=True,
                    ticklabelposition="outside",
                    tickfont=dict(size=12)
                ),

                uirevision=True
            )

            st.plotly_chart(fig, use_container_width=True)
            # height = WINDOW_ROWS * ROW_HEIGHT + 250
            # WRAP_WIDTH = 30  # nb de caractères avant retour à la ligne (ajuste)


            # y_order = df["Task"].tolist()
            

            # fig = px.timeline(
            #     df,
            #     x_start="Start",
            #     x_end="Finish",
            #     y="Task",
            #     color="Version",
            #     hover_name="Task",
            #     category_orders={"Task": y_order},   # <-- clé
            #     color_discrete_map={
            #         "NEW": "red",
            #         "OLD": "blue"
            #     }
            # )

            # fig.update_yaxes(autorange="reversed")

            # fig.update_yaxes(
            #     range=[WINDOW_ROWS, -1],
            #     fixedrange=False
            # )

            # fig.update_layout(
            #     title="Project Plan Gantt Chart",
            #     height=height,
            #     bargap=0.2,
            #     title_x=0.5,
            #     dragmode="pan",
            #     xaxis=dict(
            #         rangeslider_visible=True,
            #         side="top",
            #         showgrid=True,
            #         zeroline=True,
            #         showline=True,
            #         tickformat="%x\n"
            #     ),
            #     yaxis=dict(automargin=True),
            #     uirevision=True
            # )

            # st.plotly_chart(fig, use_container_width=True)  #Display the plotly chart in Streamlit

            

        except Exception as e:
            st.warning('You need to upload a csv file.')
            st.exception(e)

            
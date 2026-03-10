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

ID_COL = "Activity ID"  # <-- change if needed
DATE_COLS = ["Start", "Finish"]  # <-- change if needed

# Which columns to compare (only those present will be used)
COLS_TO_COMPARE = [
    "Section",
    "Activity Name",
    "Start",
    "Finish",
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
    df = df.replace(r"^\s*$", pd.NA, regex=True)
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
                .str.replace("\u00A0", " ", regex=False)  # non-breaking space
                .str.replace("\u200B", "", regex=False)   # zero-width space
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
            # nettoyer les suffixes " A" et les espaces
            df[date_col] = (
                df[date_col].astype(str)
                .str.strip()
                .str.replace(" A", "", regex=False)
            )
            # conversion en datetime (format européen)
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)



    return df


# =========================
# COMPARISON
# =========================

def compare_versions_simple(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Outer merge on ID and compute ADDED/REMOVED/MODIFIED/UNCHANGED."""

    # Keep only activity rows (need an ID to compare)
    df_old = df_old[df_old[ID_COL].notna()].copy()
    df_new = df_new[df_new[ID_COL].notna()].copy()

    # Optional filter: keep only IDs starting with X
    if FILTER_ID_STARTS_WITH:
        df_old = df_old[df_old[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()
        df_new = df_new[df_new[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()

    # Reduce to useful columns
    old_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_old.columns]
    new_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_new.columns]

    df_old = df_old[old_keep].copy()
    df_new = df_new[new_keep].copy()

    # Defensive: ensure ID is string
    df_old[ID_COL] = df_old[ID_COL].astype("string").str.strip()
    df_new[ID_COL] = df_new[ID_COL].astype("string").str.strip()

    # Merge (outer)
    merged = df_old.merge(
        df_new,
        on=ID_COL,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        # validate="one_to_one"  # enable if you are 100% sure IDs are unique
    )

    # Base status
    merged["Status"] = merged["_merge"].map({
        "left_only": "REMOVED",
        "right_only": "ADDED",
        "both": "EXISTING",
    }).astype(str)

    # Column-by-column change detection
    for col in COLS_TO_COMPARE:
        old_col = f"{col}_old"
        new_col = f"{col}_new"

        if old_col in merged.columns and new_col in merged.columns:
            # Special case: datetime comparison (ignore time)
            if pd.api.types.is_datetime64_any_dtype(merged[old_col]) or pd.api.types.is_datetime64_any_dtype(merged[new_col]):
                old_vals = pd.to_datetime(merged[old_col], errors="coerce").dt.date
                new_vals = pd.to_datetime(merged[new_col], errors="coerce").dt.date
                merged[f"{col}_changed"] = old_vals != new_vals
            else:
                merged[f"{col}_changed"] = (
                    merged[old_col].astype("string").fillna(pd.NA)
                    != merged[new_col].astype("string").fillna(pd.NA)
                )
        else:
            merged[f"{col}_changed"] = False

    change_cols = [f"{c}_changed" for c in COLS_TO_COMPARE]
    merged["Any_change"] = merged[change_cols].any(axis=1)

    merged.loc[(merged["Status"] == "EXISTING") & (merged["Any_change"]), "Status"] = "MODIFIED"
    merged.loc[(merged["Status"] == "EXISTING") & (~merged["Any_change"]), "Status"] = "UNCHANGED"

    # Sorting
    status_order = {"MODIFIED": 0, "ADDED": 1, "REMOVED": 2, "UNCHANGED": 3}
    merged["Status_sort"] = merged["Status"].map(status_order).fillna(99)
    merged = merged.sort_values(["Status_sort", ID_COL]).drop(columns=["Status_sort"])

    # Remove unchanged (as requested)
    merged = merged[merged["Status"] != "UNCHANGED"].copy()

    # Nice-to-have columns
    if "Start_changed" in merged.columns:
        merged["Start Date changed"] = merged["Start_changed"]
    if "End_changed" in merged.columns:
        merged["End Date changed"] = merged["End_changed"]

    return merged

def compare_versions_focused2(df_old: pd.DataFrame, df_new: pd.DataFrame, prio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sur ID + calcul des statuts.
    """
    # On garde uniquement les colonnes utiles
    old_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_old.columns]
    new_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_new.columns]

    df_old = df_old[old_keep].copy()
    df_new = df_new[new_keep].copy()

    priority_ids = set(
    prio_df["Activity ID"].dropna().astype(str).str.strip()
)


    # Merge outer
    merged = df_old.merge(
        df_new,
        on=ID_COL,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True
    )

    merged["Status"] = merged["_merge"].map({
        "left_only": "REMOVED",
        "right_only": "ADDED",
        "both": "EXISTING"
    }).astype(str)


    # Colonnes "changed?"
    for col in COLS_TO_COMPARE:
        old_col = f"{col}_old"
        new_col = f"{col}_new"

        if old_col in merged.columns and new_col in merged.columns:
            merged[f"{col}_changed"] = (
                merged[old_col].fillna(pd.NA).astype("object")
                != merged[new_col].fillna(pd.NA).astype("object")
            )
        else:
            merged[f"{col}_changed"] = False

    # MODIFIED = EXISTING + au moins une colonne changée
    change_cols = [f"{c}_changed" for c in COLS_TO_COMPARE]
    merged["Any_change"] = merged[change_cols].any(axis=1)

    merged.loc[(merged["Status"] == "EXISTING") & (merged["Any_change"]), "Status"] = "MODIFIED"
    merged.loc[(merged["Status"] == "EXISTING") & (~merged["Any_change"]), "Status"] = "UNCHANGED"
    
    # Colonnes pratiques
    if "Start_changed" in merged.columns:
        merged["Start Date changed"] = merged["Start_changed"]
    if "End_changed" in merged.columns:
        merged["End Date changed"] = merged["End_changed"]
    
    # Tri : d’abord MODIFIED / ADDED / REMOVED
    status_order = {"MODIFIED": 0, "ADDED": 1, "REMOVED": 2, "UNCHANGED": 3}
    merged["Status_sort"] = merged["Status"].map(status_order).fillna(99)

    
    merged = merged.sort_values(["Status_sort", ID_COL]).drop(columns=["Status_sort"])
    
    # ✅ Enlève les lignes UNCHANGED
    merged = merged[merged["Status"] != "UNCHANGED"].copy()

    merged["is_priority"] = merged["Activity ID"].astype(str).str.strip().isin(priority_ids)
    prio_changes = merged[merged["is_priority"]].copy()
    other_changes = merged[~merged["is_priority"]].copy()

    return merged, prio_changes, other_changes




def compare_versions_focused(df_old: pd.DataFrame, df_new: pd.DataFrame, prio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sur ID + calcul des statuts.
    """
    # Keep only activity rows (need an ID to compare)
    df_old = df_old[df_old[ID_COL].notna()].copy()
    df_new = df_new[df_new[ID_COL].notna()].copy()

    # Optional filter: keep only IDs starting with X
    if FILTER_ID_STARTS_WITH:
        df_old = df_old[df_old[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()
        df_new = df_new[df_new[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()
        prio_df = prio_df[prio_df[ID_COL].astype(str).str.match(fr"^{FILTER_ID_STARTS_WITH}", na=False)].copy()


    # Reduce to useful columns
    old_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_old.columns]
    new_keep = [ID_COL] + [c for c in COLS_TO_COMPARE if c in df_new.columns]

    df_old = df_old[old_keep].copy()
    df_new = df_new[new_keep].copy()

    priority_ids = set(
    prio_df["Activity ID"].dropna().astype(str).str.strip()
)
    # Defensive: ensure ID is string
    df_old[ID_COL] = df_old[ID_COL].astype("string").str.strip()
    df_new[ID_COL] = df_new[ID_COL].astype("string").str.strip()

    # Merge (outer)
    merged = df_old.merge(
        df_new,
        on=ID_COL,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        # validate="one_to_one"  # enable if you are 100% sure IDs are unique
    )

    # Base status
    merged["Status"] = merged["_merge"].map({
        "left_only": "REMOVED",
        "right_only": "ADDED",
        "both": "EXISTING",
    }).astype(str)


    # Column-by-column change detection
    for col in COLS_TO_COMPARE:
        old_col = f"{col}_old"
        new_col = f"{col}_new"

        if old_col in merged.columns and new_col in merged.columns:
            # Special case: datetime comparison (ignore time)
            if pd.api.types.is_datetime64_any_dtype(merged[old_col]) or pd.api.types.is_datetime64_any_dtype(merged[new_col]):
                old_vals = pd.to_datetime(merged[old_col], errors="coerce").dt.date
                new_vals = pd.to_datetime(merged[new_col], errors="coerce").dt.date
                merged[f"{col}_changed"] = old_vals != new_vals
            else:
                merged[f"{col}_changed"] = (
                    merged[old_col].astype("string").fillna(pd.NA)
                    != merged[new_col].astype("string").fillna(pd.NA)
                )
        else:
            merged[f"{col}_changed"] = False

    # MODIFIED = EXISTING + au moins une colonne changée
    change_cols = [f"{c}_changed" for c in COLS_TO_COMPARE]
    merged["Any_change"] = merged[change_cols].any(axis=1)

    merged.loc[(merged["Status"] == "EXISTING") & (merged["Any_change"]), "Status"] = "MODIFIED"
    merged.loc[(merged["Status"] == "EXISTING") & (~merged["Any_change"]), "Status"] = "UNCHANGED"
    
    # Colonnes pratiques
    if "Start_changed" in merged.columns:
        merged["Start Date changed"] = merged["Start_changed"]
    if "End_changed" in merged.columns:
        merged["End Date changed"] = merged["End_changed"]
    
    # Tri : d’abord MODIFIED / ADDED / REMOVED
    status_order = {"MODIFIED": 0, "ADDED": 1, "REMOVED": 2, "UNCHANGED": 3}
    merged["Status_sort"] = merged["Status"].map(status_order).fillna(99)

    
    merged = merged.sort_values(["Status_sort", ID_COL]).drop(columns=["Status_sort"])
    
    # ✅ Enlève les lignes UNCHANGED
    merged = merged[merged["Status"] != "UNCHANGED"].copy()

    merged["is_priority"] = merged["Activity ID"].astype(str).str.strip().isin(priority_ids)
    prio_changes = merged[merged["is_priority"]].copy()
    other_changes = merged[~merged["is_priority"]].copy()

    return merged, prio_changes, other_changes

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
        "Task": df["Activity Name_old"].astype(str) + " OLD",
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
            excel_bytes = to_excel_bytes(df_summary, df_comparison)
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
            prio_df = pd.read_excel(focused_ID)

            df_old = clean_df(df_old, id_col=ID_COL)
            df_new = clean_df(df_new, id_col=ID_COL)
            prio_df = clean_df(prio_df, id_col=ID_COL)

            # Guard: ID column must exist
            if ID_COL not in df_old.columns or ID_COL not in df_new.columns:
                st.error(f"Column '{ID_COL}' must exist in BOTH files.")
                st.stop()

            # Compare
            df_merged, df_prio_changes, df_other_changes = compare_versions_focused(df_old, df_new, prio_df)

            # Optionally remove unchanged in UI (already removed by default in compare_versions)
            if not remove_unchanged:
                pass

            # Summary
            df_summary = (
                df_merged["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
            )

            # KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("Modified", int((df_merged["Status"] == "MODIFIED").sum()))
            k2.metric("Added", int((df_merged["Status"] == "ADDED").sum()))
            k3.metric("Removed", int((df_merged["Status"] == "REMOVED").sum()))

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
            st.dataframe(df_prio_changes, use_container_width=True, height=520)

            # Download
            excel_bytes = to_excel_bytes(df_summary, df_prio_changes, df_other_changes)
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

            Tasks = duplicate_new_old_rows(df_prio_changes.copy())

            Tasks['Start'] = pd.to_datetime(Tasks['Start'])
            Tasks['Finish'] = pd.to_datetime(Tasks['Finish'])

            with st.expander("📋 View generated tasks", expanded=False):
                st.dataframe(
                    Tasks,
                    use_container_width=True,
                    height=400
                )
            Tasks=duplicate_new_old_rows(df_prio_changes.copy())

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

            
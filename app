"""
Mapping Sheet Automation - Streamlit app

Run:
  pip install streamlit pandas openpyxl requests pillow torch transformers
  streamlit run app.py

(make_mapping_sheet.py isi folder me hona chahiye)
"""
import io
import pandas as pd
import streamlit as st

import make_mapping_sheet as mm

st.set_page_config(page_title="Mapping Sheet Automation", layout="wide")
st.title("Mapping Sheet Automation")
st.caption("Label to group file + CGC file (class images) se image-based mapping sheet")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    parent_col = st.text_input("Parent group column (label file)", value="A",
                               help="Jis column me Parent group likha hai, uska letter. Child uske aage ke columns me hote hain.")
    high = st.slider("HIGH: is score ya upar = auto match", 0.50, 1.00, mm.HIGH, 0.01)
    low = st.slider("LOW: is score se neeche = others", 0.30, 1.00, mm.LOW, 0.01)
    if low > high:
        st.error("LOW, HIGH se chhota hona chahiye")
    st.markdown("Beech ke score wali rows **REVIEW** me aati hain.")

# ------------------------------------------------------------------ uploads
c1, c2 = st.columns(2)
with c1:
    label_file = st.file_uploader("1. Label to group file (.xlsx)", type=["xlsx"])
with c2:
    cgc_files = st.file_uploader(
        "2. CGC file(s) (.xlsx) - parent aur child dono ki images",
        type=["xlsx"], accept_multiple_files=True,
        help="Columns: category_name, group_name, class_name, class_image_gcs_file_path. "
             "Parent (ABC) aur child ki images alag files me hon to dono upload kar dein.")

if label_file:
    label_df = mm.read_label_file(label_file, parent_col)
    with st.expander(f"Label file preview ({len(label_df)} parent-child rows)"):
        st.dataframe(label_df, use_container_width=True, hide_index=True)

cgc = None
if cgc_files:
    cgc = pd.concat([pd.read_excel(f) for f in cgc_files], ignore_index=True)
    need = {"group_name", "class_name", "class_image_gcs_file_path"}
    if not need.issubset(cgc.columns):
        st.error(f"CGC file me ye columns chahiye: {sorted(need)}")
        cgc = None
    else:
        cgc = cgc.dropna(subset=["class_image_gcs_file_path"]).drop_duplicates(
            subset=["group_name", "class_name", "class_image_gcs_file_path"])
        st.caption(f"CGC: {len(cgc)} images, {cgc['group_name'].nunique()} groups")

# ------------------------------------------------------------------ run
if st.button("Mapping sheet banao", type="primary",
             disabled=not (label_file and cgc is not None and low <= high)):
    progress = st.progress(0.0, text="Images download + embeddings...")

    def embed_with_progress(urls, batch=32):
        import numpy as np
        parts = []
        for i in range(0, len(urls), batch):
            parts.append(mm.embed_urls(urls[i:i + batch], batch))
            progress.progress(min((i + batch) / len(urls), 1.0),
                              text=f"Images processed: {min(i + batch, len(urls))}/{len(urls)}")
        return np.vstack(parts) if parts else np.zeros((0, 512), dtype="float32")

    try:
        result = mm.build_mapping(label_df, cgc, embed_fn=embed_with_progress, high=high, low=low)
        progress.empty()
        st.session_state["result"] = result
    except Exception as e:
        progress.empty()
        st.error(f"Error: {e}")

# ------------------------------------------------------------------ result
if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Result")

    counts = result["Status"].value_counts()
    cols = st.columns(4)
    for col, name in zip(cols, ["AUTO-MATCH", "AUTO-OTHERS", "REVIEW", "NO IMAGE"]):
        col.metric(name, int(counts.get(name, 0)))

    show = st.multiselect("Status filter", sorted(result["Status"].unique()),
                          default=sorted(result["Status"].unique()))
    st.caption("Table editable hai: REVIEW rows me Parent group / Parent Class khud theek kar sakte hain.")
    edited = st.data_editor(result[result["Status"].isin(show)], use_container_width=True,
                            hide_index=True, num_rows="fixed", key="editor")

    # edits ko poore result me wapas merge karo
    final = result.copy()
    final.loc[edited.index, edited.columns] = edited

    buf = io.BytesIO()
    mm.save(final, buf)
    st.download_button("Excel download karo (mapping_sheet.xlsx)", buf.getvalue(),
                       file_name="mapping_sheet.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

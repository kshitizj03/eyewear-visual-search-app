"""Streamlit demo UI for the Eyewear Visual Search platform.

Talks directly to the in-process SearchEngine (no HTTP hop needed for the demo),
so it works even if the API isn't running. Shows: upload, optional text modifier,
structured filters, a results grid with similarity score + score breakdown +
predicted tags, and Relevant / Not-Relevant feedback buttons that re-rank live.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# make `src` importable when run via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feedback.feedback import FeedbackService  # noqa: E402
from src.pipeline.search import SearchEngine        # noqa: E402

st.set_page_config(page_title="Eyewear Visual Search", page_icon="🕶️", layout="wide")


@st.cache_resource(show_spinner="Loading CLIP model + FAISS index …")
def load_engine() -> SearchEngine:
    return SearchEngine()


@st.cache_resource
def load_feedback() -> FeedbackService:
    return FeedbackService()


st.title("🕶️ Eyewear Visual Search")
st.caption("Upload a photo of glasses (or someone wearing them) to find visually similar frames. "
           "Ranking fuses CLIP style similarity + colour + attribute tags + your feedback.")

try:
    engine = load_engine()
    feedback = load_feedback()
except FileNotFoundError:
    st.error("No index found. Build it first:  `python -m scripts.build_index`")
    st.stop()

# ------------------------------------------------------------------ #
# Sidebar: query + filters
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("Query")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    text_mod = st.text_input("Text modifier (optional)", placeholder="but in tortoise shell")
    smart = st.checkbox("Smart crop (detect face -> glasses)", value=True)

    st.header("Filters")
    fl = engine.db
    lo, hi = fl.price_bounds()
    price = st.slider("Price range", float(lo), float(max(hi, lo + 1)),
                      (float(lo), float(hi)))
    brand = st.selectbox("Brand", ["Any"] + fl.distinct("brand"))
    material = st.selectbox("Material", ["Any"] + fl.distinct("material"))
    top_k = st.slider("Results", 3, 24, 12)
    go = st.button("🔍 Search", type="primary", use_container_width=True)

# ------------------------------------------------------------------ #
# Run search
# ------------------------------------------------------------------ #
if go and uploaded:
    image = Image.open(uploaded).convert("RGB")
    resp = engine.search(
        image, text=text_mod or None, top_k=top_k,
        price_min=price[0], price_max=price[1],
        brand=None if brand == "Any" else brand,
        material=None if material == "Any" else material,
        use_smart_crop=smart,
    )
    st.session_state["resp"] = resp
    st.session_state["query_image"] = image

resp = st.session_state.get("resp")
if resp:
    left, right = st.columns([1, 3])
    with left:
        st.image(st.session_state["query_image"], caption="Your query", use_container_width=True)
        st.metric("Latency", f"{resp.latency_ms} ms")
        if resp.cropped:
            st.success("Smart-crop applied ✂️")
        st.subheader("Predicted attributes")
        for group, res in resp.query_tags.items():
            st.write(f"**{group}:** {res['label']}  ·  {res['confidence']:.0%}")

    with right:
        st.subheader(f"{len(resp.results)} similar products")
        cols = st.columns(3)
        for i, r in enumerate(resp.results):
            with cols[i % 3]:
                img_path = r.metadata.get("image_path")
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                name = r.metadata.get("name") or r.product_id
                st.markdown(f"**{name}**")
                st.caption(
                    f"{r.metadata.get('brand','')} · {r.metadata.get('material','')} · "
                    f"₹{r.metadata.get('price','')}"
                )
                st.progress(min(1.0, max(0.0, r.similarity_score)),
                            text=f"Similarity {r.similarity_score:.3f}")
                with st.expander("Score breakdown"):
                    st.json(r.components)
                b1, b2 = st.columns(2)
                style = resp.query_tags.get("style", {}).get("label", "")
                if b1.button("👍 Relevant", key=f"rel_{r.product_id}_{i}"):
                    out = feedback.record(resp.query_id, r.product_id, style, True)
                    st.toast(f"Boosted -> {out['boost']:+.3f}")
                if b2.button("👎 Not", key=f"not_{r.product_id}_{i}"):
                    out = feedback.record(resp.query_id, r.product_id, style, False)
                    st.toast(f"Demoted -> {out['boost']:+.3f}")
elif go and not uploaded:
    st.warning("Please upload an image first.")

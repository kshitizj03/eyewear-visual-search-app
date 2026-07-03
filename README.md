# Eyewear Visual Search

An AI powered visual similarity search platform for eyewear. A user uploads a photo of a
pair of glasses, or a photo of someone wearing glasses, and the system returns the most
visually similar frames from the catalog. Results are ranked by style, colour, shape and
material, with structured filters (price, brand, material) and a feedback loop that learns
from user clicks.

Built for the Lenskart PS II assignment "Visual Similarity Search for Eyewear".

## Table of contents

1. [What it does](#what-it-does)
2. [Why these design choices](#why-these-design-choices)
3. [Architecture](#architecture)
4. [Project layout](#project-layout)
5. [Quickstart](#quickstart)
6. [API reference](#api-reference)
7. [How the feedback loop works](#how-the-feedback-loop-works)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Scaling notes](#scaling-notes)
11. [Deliverables map](#deliverables-map)
12. [Known limitations](#known-limitations)

## What it does

| Requirement | How it is met |
|---|---|
| Image ingestion and feature extraction | A reusable pipeline: preprocess, then CLIP embedding, colour histogram, zero shot tags, then store |
| Vector search | FAISS exact cosine index (`IndexFlatIP`) over L2 normalized CLIP embeddings |
| Multi attribute similarity | A fusion ranker: `0.70 * CLIP + 0.18 * colour + 0.12 * attribute overlap + feedback boost` |
| Structured filters | Price range, brand and material, applied in SQLite |
| Attribute recognition (mandatory 3.3) | Zero shot classification via CLIP text prompts (aviator, wayfarer, round, rimless, metal, acetate, transparent and more) |
| Feedback loop (mandatory 3.4) | Relevant and Not relevant clicks are logged; a per (style, product) boost is folded into ranking |
| Architecture separation | The AI inference layer (`src/models`, `src/pipeline`) is cleanly split from the data storage layer (`src/storage`) |
| Observability | Structured logging plus latency timing; failed uploads and slow queries are logged |
| Bonus: smart cropping | OpenCV face detection crops to the frame region for on model or selfie photos, and falls back to the full image otherwise |
| Bonus: multi modal search | Upload a black frame and type "tortoise shell" to blend the image and text embeddings |

## Why these design choices

### Model: CLIP (ViT B/32), a Vision Transformer

CLIP is used as a single backbone for three jobs: image embeddings, text embeddings (for
tagging and multi modal search), and a shared embedding space so the two are directly
comparable.

Why CLIP instead of a plain ResNet or ImageNet CNN: a ResNet trained on ImageNet produces
features tuned to 1000 object classes, and "aviator versus wayfarer" is not in that
vocabulary, so its embeddings cluster by generic object cues. CLIP is trained on more than
400 million image and text pairs, so its space is semantically organised. Visually and
stylistically similar eyewear lands close together, and the same space can be queried and
tagged with natural language for free.

ViT versus CNN, measured rather than asserted: a runnable ResNet50 baseline is included in
`scripts/compare_models.py`. On this catalog, leave one out style precision at 5 is CLIP ViT
0.410 versus ResNet50 0.388, a relative gain of 5.6 percent. The ViT backbone captures the
global frame shape through self attention rather than only local receptive fields, which
suits overall silhouette similarity. Re run the script to reproduce the number.

### Distance metric: cosine similarity

All embeddings are L2 normalized, so the cosine of the angle between two vectors equals their
dot product. The system therefore uses a FAISS inner product index (`IndexFlatIP`) as an
exact cosine index. Cosine is the standard choice for CLIP because it compares direction
(semantic content) and ignores magnitude, which makes it more robust to lighting and contrast
than Euclidean distance. Euclidean distance would additionally penalise differences in vector
length that carry little meaning here, so it was not chosen.

### Vector database: FAISS with SQLite

FAISS was chosen from the assignment options (Pinecone, Milvus, FAISS) because it is fully
self contained (no external service and no API key), exact at this catalog size, and answers
in well under a millisecond. Human readable metadata and filters live in SQLite. Keeping the
vector store and the structured store separate is exactly the "clear separation between AI
inference and data storage" the non functional requirements ask for.

### Ranking: multi attribute fusion (the accuracy lever)

Pure nearest neighbour on a single embedding under weights colour and exact attributes. The
engine over fetches `TOP_K * 6` candidates from FAISS, applies the structured filters, then re
ranks with a weighted blend of CLIP similarity, HSV colour histogram intersection, and
predicted tag overlap, plus the learned feedback boost. The weights live in `src/config.py`
and are shown per result in the UI under "Score breakdown", so the ranking is fully
transparent.

Two details make the secondary signals genuinely useful:

* Background aware colour. Product shots share a white studio backdrop that would make every
  colour histogram look identical. `color_features.py` masks low saturation and high value
  background pixels so the histogram reflects the frame colour. A black frame now scores much
  higher against other black frames than against tortoise ones.
* Prompt ensembled tagging. Each attribute label is scored against several text templates
  whose embeddings are averaged into one prototype (the standard CLIP zero shot trick), which
  gives more stable tags than a single caption.

### Preprocessing and robust image handling (requirement 3.1)

Every image passes through one shared path so that catalog images and query images are
processed identically:

* Orientation and mode (`utils/image.py:to_rgb`): EXIF auto rotate, transparency flattened
  onto white (RGBA, palette PNG, WebP), mode converted to RGB.
* Colour correction (`utils/image.py:color_correct`): gray world white balance to neutralise
  lighting casts. It is close to identity on clean product shots and helps on model queries.
* Resize and centre crop: handled by the CLIP preprocessing transform at 224 by 224.

Inputs in any common format are accepted. `scripts/convert_images.py` normalises a folder to
JPEG or PNG (the standard formats named in the assignment), and the demo dataset ships as
JPEG.

## Architecture

The rendered diagram is `architecture.svg` (open it in any browser). The editable source is
`architecture.mmd` (paste into https://mermaid.live to re export).

```
Upload (plus optional text)
        |
        v
 AI inference layer: smart crop, colour correct, CLIP image encode,
 colour histogram, zero shot tags, (optional) CLIP text encode, fuse query vector
        |
        v
 Data storage layer: FAISS cosine ANN  +  SQLite (metadata, colour, tags, feedback)
        |
        v
 Ranking: structured filters (price, brand, material) then fusion re rank (plus feedback boost)
        |
        v
 Serving: FastAPI and Streamlit  ->  ranked products with similarity score and tags
        ^                                        |
        |________ user 'relevant' clicks ________|
```

## Project layout

```
src/
  config.py               # every tunable knob (model, weights, top_k, prompts)
  models/                 # AI inference layer
    embedder.py           #   CLIP wrapper (image and text) plus ResNet50 CNN baseline
    color_features.py     #   background aware HSV colour histogram and similarity
    attribute_tagger.py   #   prompt ensembled zero shot attribute classification
    smart_crop.py         #   face to frame region crop (bonus)
  storage/                # data storage layer
    vector_store.py       #   FAISS index (build, save, load, search)
    metadata_db.py        #   SQLite: products, feedback, boosts, filters
  pipeline/
    ingest.py             #   reusable ingestion pipeline
    search.py             #   query plus multi attribute fusion ranking
  feedback/feedback.py    #   feedback loop service
  api/main.py             # FastAPI app (Swagger at /docs)
  utils/
    image.py              #   robust loader (EXIF, transparency, RGB, colour correct)
    logging.py            #   logging and latency timer
app/streamlit_app.py      # demo UI
scripts/
  collect_data.py         # build the sample dataset (--download or --from-folder)
  convert_images.py       # normalise a folder to JPEG or PNG
  build_index.py          # run ingestion into FAISS plus SQLite
  compare_models.py       # CNN (ResNet50) versus ViT (CLIP) retrieval comparison
tests/test_pipeline.py    # fast unit tests (no model download needed)
```

## Quickstart

### 1. Install

Use Python 3.11 or 3.12. Python 3.13 and 3.14 do not yet have stable wheels for some
dependencies.

```powershell
cd eyewear-visual-search-app
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Verified stack: Python 3.11, torch 2.12, open-clip 3.3, faiss-cpu 1.14, opencv 4.13.
`opencv-python` is pinned below version 5 because OpenCV 5.x dropped the bundled Haar
cascades used by smart crop. On a GPU, install a CUDA build of PyTorch first (see
pytorch.org), then the rest; `faiss-cpu` is fine for a few thousand images.

### 2. Get a dataset

Option A, bring your own (recommended and fastest): drop eyewear images (JPG, PNG or WebP)
into `data/images/`, then run:

```bash
python -m scripts.collect_data --from-folder
```

Metadata is resolved from the filename. Descriptive Lenskart names, for example
`black-full-rim-square-vincent-chase-acetate-vc-e18174-eyeglasses.jpg`, are parsed for brand,
shape, material and colour. Anything missing is filled with CLIP zero shot tags. For full
control you can instead use explicit `key=value` names, for example
`brand=RayBan__style=aviator.jpg`.

Option B, download from URLs: put product image URLs in `data/seeds.csv`, then run:

```bash
python -m scripts.collect_data --download
```

Optional, normalise everything to the standard formats, for example WebP to JPEG (originals
are backed up to `data/images_original/`):

```bash
python -m scripts.convert_images --to jpeg   # run before collect_data
```

### 3. Build the index

```bash
python -m scripts.build_index
```

Optional, compare the CNN baseline against the ViT backbone:

```bash
python -m scripts.compare_models
```

### 4. Run

```bash
# demo UI
streamlit run app/streamlit_app.py

# or the API (Swagger docs at http://localhost:8000/docs)
uvicorn src.api.main:app --reload
```

### 5. Test

```bash
pytest -q
```

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/search` | Multipart image plus optional `text`, `top_k`, `price_min`, `price_max`, `brand`, `material`, `smart_crop`. Returns ranked results with a similarity score and predicted tags. |
| POST | `/feedback` | Body `{query_id, product_id, style, relevant}`. Records the click and returns the updated boost. |
| GET | `/product/{id}` | Product metadata. |
| GET | `/filters` | Available brands, materials, styles and the price bounds, for the UI. |
| GET | `/health` | Status, device and indexed count. |

Example:

```bash
curl -X POST http://localhost:8000/search \
  -F "file=@data/images/sample.jpg" -F "text=tortoise shell" -F "top_k=8"
```

Each result carries a `components` object with the `clip`, `colour`, `attr` and `boost` parts
of its score, so the ranking is auditable.

## How the feedback loop works

When a user marks a result Relevant or Not relevant, the click is stored in the SQLite
`feedback` table together with the predicted style of the query. A running score in the
`boosts` table is updated per (style, product), bounded so feedback nudges the ranking rather
than dominating it. On the next search for that visual style, the boost is added into the
fusion score, so frequently approved products for a given style surface higher over time. This
is a simple, transparent and fully local implementation of learning from interaction.

## Configuration

All tunable values live in `src/config.py`:

* `CLIP_MODEL_NAME` and `CLIP_PRETRAINED`: the backbone (ViT B/32 by default; ViT L/14 for
  more accuracy on a GPU).
* `FUSION_WEIGHTS`: the blend of CLIP, colour and attribute signals.
* `TOP_K` and `CANDIDATE_MULTIPLIER`: how many results to return and how many to over fetch
  before re ranking.
* `ATTRIBUTE_PROMPTS`: the zero shot vocabulary for style, rim, material and transparency.
* `FEEDBACK_BOOST_PER_CLICK` and `FEEDBACK_BOOST_CAP`: how strongly feedback affects ranking.
* `SLOW_QUERY_MS`: the latency threshold above which a query is logged as slow.

## Testing

`pytest -q` runs fast unit tests that do not require downloading the CLIP weights. They cover
colour similarity bounds, the FAISS cosine ordering, SQLite metadata filtering, feedback boost
accumulation with capping, the robust image loader, colour correction and the background aware
histogram. One optional test exercises the real CLIP text encoder and is skipped automatically
if the weights are not available.

## Scaling notes

The catalog here is small, so `IndexFlatIP` is exact and instant. For a production catalog of
millions of vectors, swap `IndexFlatIP` for `IndexHNSWFlat` or `IndexIVFFlat` in
`storage/vector_store.py`. No calling code changes, because the store exposes the same
`add`, `save`, `load` and `search` interface. The SQLite metadata store can likewise be
swapped for Postgres behind the same methods.

## Deliverables map

1. Source code: this repository (clean, documented and tested).
2. Architecture diagram: `architecture.svg` (rendered) plus `architecture.mmd` (editable
   source).
3. README: this file (model choice, cosine versus Euclidean rationale, run steps).
4. Sample dataset: `data/images/` (product shots) plus `data/catalog.csv`.
   `data/demo_queries/` holds on model photos for the smart crop demo.
5. Demo video: 5 to 10 minutes; a scene outline is provided separately.

## Known limitations

* Attribute tags come from zero shot CLIP, so visually close shapes such as round and oval can
  be confused at low confidence. The confidence is reported, and shape is only a secondary
  ranking signal.
* Smart crop uses a frontal face cascade, so strongly angled or profile faces are not detected
  and fall back to full image search.
* On model query photos include skin and background, so their scores are lower than clean
  product shots. This is an expected domain gap and is stated honestly rather than hidden.
* Brand is treated as a filter and as metadata rather than as a visual similarity signal,
  because two visually identical frames from different brands should still match.

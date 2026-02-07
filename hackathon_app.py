import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from shapely import wkt
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import explain_validity
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

APP_TITLE = "Hackathon QA - Single Error Type"
ERROR_TYPE = "Self-Intersection"

BASE_DIR = Path(__file__).parent
TRAIN_GOOD = BASE_DIR / "training" / "good_examples.wkt"
ERROR_EXAMPLES = BASE_DIR / "errors" / "self_intersection_examples.wkt"

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.caption(
    "Option B: Rule-Based Geometry Validation | "
    "Single error type: self-intersecting polygons"
)


def load_wkts_from_text(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = []
    parse_errors = []
    for idx, line in enumerate(lines, 1):
        try:
            geom = wkt.loads(line)
            rows.append({"id": idx, "wkt": line, "geom": geom})
        except Exception as exc:
            parse_errors.append({"id": idx, "wkt": line, "error": str(exc)})
    return rows, parse_errors


def geom_features(geom):
    if geom.is_empty:
        return None
    area = geom.area
    perimeter = geom.length
    minx, miny, maxx, maxy = geom.bounds
    width = max(maxx - minx, 0.0)
    height = max(maxy - miny, 0.0)
    aspect = width / height if height > 0 else 0.0
    if isinstance(geom, Polygon):
        vertex_count = len(list(geom.exterior.coords))
    elif isinstance(geom, LineString):
        vertex_count = len(list(geom.coords))
    else:
        vertex_count = 1
    compactness = 0.0
    if area > 0 and perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter ** 2)
    return [area, perimeter, vertex_count, aspect, compactness]


def train_model(good_rows):
    feats = []
    for row in good_rows:
        f = geom_features(row["geom"])
        if f is not None:
            feats.append(f)
    if len(feats) < 3:
        return None, None
    X = np.array(feats, dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(Xs)
    return model, scaler


def self_intersection_info(geom):
    msg = explain_validity(geom)
    if msg is None:
        return False, ""
    if "Self-intersection" in msg or "self-intersection" in msg:
        loc = ""
        m = re.search(r"\[(.+)\]", msg)
        if m:
            loc = m.group(1)
        return True, loc
    return False, ""


with st.sidebar:
    st.header("Inputs")
    source = st.radio(
        "Choose input source",
        ["Upload WKT file", "Use provided error examples"],
        index=1,
    )
    st.markdown("---")
    st.markdown("Training data")
    st.code(str(TRAIN_GOOD), language="text")

st.markdown("## 1) Load Data")

if source == "Upload WKT file":
    uploaded = st.file_uploader("Upload WKT (.wkt)", type=["wkt", "txt"])
    input_text = uploaded.read().decode("utf-8") if uploaded else ""
else:
    input_text = ERROR_EXAMPLES.read_text(encoding="utf-8")

if not input_text:
    st.info("Provide a WKT file to begin.")
    st.stop()

rows, parse_errors = load_wkts_from_text(input_text)

if parse_errors:
    st.warning("Some lines failed to parse. They will be skipped.")
    st.json(parse_errors)

if not rows:
    st.error("No valid geometries found.")
    st.stop()

st.markdown("## 2) Train Baseline (Good Examples)")
if TRAIN_GOOD.exists():
    good_rows, good_errors = load_wkts_from_text(TRAIN_GOOD.read_text(encoding="utf-8"))
    if good_errors:
        st.warning("Training file has parse errors; those lines are ignored.")
    model, scaler = train_model(good_rows)
    if model is None:
        st.warning("Not enough good examples to train ML model. Using rule-only mode.")
else:
    model, scaler = None, None
    st.warning("Training file not found. Using rule-only mode.")

st.markdown("## 3) Detect Single Error Type")

report = []
for row in rows:
    geom = row["geom"]
    if not isinstance(geom, Polygon):
        continue
    is_err, loc = self_intersection_info(geom)
    score = None
    if model is not None:
        f = geom_features(geom)
        if f is not None:
            X = scaler.transform([f])
            score = float(model.decision_function(X)[0])
    if is_err:
        report.append({
            "id": row["id"],
            "error_type": ERROR_TYPE,
            "location": loc or "unknown",
            "ml_score": score,
            "wkt": row["wkt"],
        })

if not report:
    st.success("No self-intersection errors detected in polygon geometries.")
else:
    df = pd.DataFrame(report)
    st.error(f"Detected {len(df)} self-intersection error(s)")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Simple Error Report")
    json_report = json.dumps(report, indent=2)
    st.download_button("Download JSON Report", data=json_report, file_name="error_report.json")

st.markdown("## 4) Demo (2-3 Provided Errors)")
st.write("Using the included examples file:")
st.code(str(ERROR_EXAMPLES), language="text")

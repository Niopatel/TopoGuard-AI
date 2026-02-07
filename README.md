# TOPO GUARD - Ultra Power Geometry QA

This is a standalone copy of the Ultra Power Streamlit UI with Vision, Vector QA, ML, AI auto-fix,
and a Hackathon single-error section all in one app.

## Quick start (Windows)
1) Run `run.bat`
2) Open the Streamlit URL printed in the console

## Hackathon mode (Problem 2 compliance)
The main app already includes a dedicated section for single-error detection
(**self-intersection**, Option B). It uses:
- `training\good_examples.wkt`
- `errors\self_intersection_examples.wkt`

See `TRAINING.md` for how to add more examples.

## Files
- `ultra_power_app.py` - full Streamlit app
- `hackathon_app.py` - legacy single-error app (optional)
- `training\good_examples.wkt` - good examples for ML baseline
- `errors\self_intersection_examples.wkt` - 2-3 provided error examples
- `sample_data.wkt` - sample input for full app
- `complex_geometries_test.wkt` - larger sample input
- `requirements.txt` - Python dependencies

## Portable version
See `portable\` for a self-contained, movable copy with its own `run_portable.bat`.

## 🚀 Project Overview

TopoGuard AI is a lightweight AI-powered system that automatically detects **self-intersecting polygons** in vector geometry data.

In GIS and cartographic pipelines, geometry errors can break maps and analyses.  
Manual QA is slow and error-prone.

TopoGuard AI automates detection of a critical topology error using:

- Rule-based geometry validation  
- Lightweight machine learning  
- Clear and simple reporting

Built for hackathon evaluation but practical for real-world use.

---

## 🎯 Hackathon Focus

This solution strictly follows hackathon requirements:

✔ Single error type: **Self-intersection**  
✔ Rule-based validation  
✔ Simple ML anomaly detection  
✔ Clear error output  
✔ Training using good vs bad examples  
✔ Demo on provided examples

---

## 🧠 How It Works

### Step 1 — Input  
User uploads WKT (Well-Known Text) geometry data.

### Step 2 — Geometry Parsing  
WKT is converted into geometric objects.

### Step 3 — Rule-Based Validation  
Shapely detects self-intersections and invalid topology.

### Step 4 — ML Baseline  
Isolation Forest learns normal geometry patterns from valid data.

### Step 5 — Reporting  
System outputs:
- Error type  
- Error location  
- JSON report download

---

## 🧩 Features

✔ Detects self-intersecting polygons  
✔ Lightweight ML anomaly detection  
✔ JSON error reports  
✔ Simple Streamlit UI  
✔ Easy training data extension  
✔ Hackathon-focused design

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Shapely  
- Scikit-learn (Isolation Forest)  
- NumPy  
- Pandas  


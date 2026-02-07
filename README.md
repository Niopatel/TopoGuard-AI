# codexai - Ultra Power Geometry QA

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

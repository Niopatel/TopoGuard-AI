# Training Instructions (Hackathon Mode)

This project focuses on a single error type: **self-intersecting polygons**.

## Training data files
- `training\good_examples.wkt`: valid geometries used to learn normal feature ranges
- `errors\self_intersection_examples.wkt`: known bad examples (self-intersections)

## How to add more examples
1) Click **Auto Add Examples (Good + Bad)** in the app to populate both files instantly.
2) Or manually add valid geometries to `training\good_examples.wkt` (one WKT per line).
3) Add more self-intersection examples to `errors\self_intersection_examples.wkt`.
4) Restart the app to reload the training set.

## Notes
- The app uses the good examples to train a simple Isolation Forest model.
- Only **self-intersection** errors are reported; other issues are ignored.

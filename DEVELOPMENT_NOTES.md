# Development Notes (Full History and Ideation)

## 1. Purpose
This document captures the end-to-end development history of the OLED FCB edge
detection system, including trials, failures, ideation, and rationale behind the
final architecture. It is intended to help future maintainers quickly
understand why each decision was made.

## 2. Background and Early Ideation
The initial requirement was a reliable, offline edge detector for OLED FCB
side-view images, with batch processing, GUI control, and export of edge
coordinates.

The early ideation considered:
- Canny and LoG style detectors (robust, but parameter-sensitive and slower).
- Active contours/snakes (accurate but heavier and harder to stabilize).
- Classical morphology + skeletonization (lightweight, controllable).
- Lightweight auto-optimization instead of heavy ML due to offline constraints.

We chose a Sobel-based pipeline with strong pre/post-processing because it
balanced speed, controllability, and low dependency footprint.

## 3. Phase History (Key Milestones)

### Phase 0: Baseline Sobel
- Implemented Sobel gradient, NMS, double-thresholding, hysteresis.
- Observed: thin edges often broken, inner edges occasionally selected.

### Phase 1: Performance Stabilization
- Replaced Python loops with NumPy vectorization.
- Introduced BFS-based hysteresis for consistent connectivity.
- Achieved sub-second processing per image in typical cases.

### Phase 2: Recall vs Thickness Tradeoff
- Relaxed NMS to reduce gaps -> improved recall but thickened edges.
- Added thinning (Zhang-Suen) -> restored 1-pixel edges but caused inward curl
  on ambiguous gradients.

### Phase 3: Intrusion Mitigation
- Added polarity filter to remove inner edges based on gradient direction.
- Added boundary band filter using Otsu mask and morphological closing.
- Reduced inner intrusion significantly but required careful tuning for
  low-contrast images.

### Phase 4: Low-Quality Image Support
- Added auto-threshold scaling based on image contrast.
- Added median filter and contrast stretch for noisy/low-contrast inputs.
- Added soft linking to reconnect faint edges.

### Phase 5: Auto-Optimization
- Implemented coarse -> refine search with scoring.
- Added adaptive sampling and ROI clustering for speed.
- Introduced multi-metric scoring with explicit priority ordering.

### Phase 6: Usability and Visualization
- Added ROI editor with cache and multi-image navigation.
- Added real-time graphs and progress logging.
- Added zoomable graphs and score display scaling.

### Phase 7: Wrinkle and Break Detection
- Added endpoint, wrinkle, and branch metrics to penalize jagged contours.
- Added edge smoothing and spur pruning options.

## 4. Trial and Error Summary
- NMS relax helped continuity but created thicker edges.
- Thinning fixed thickness but sometimes increased inner curling.
- Boundary band reduced curling but needed mask stabilization to avoid drift.
- Soft linking recovered faint edges but could add false connections.
- Edge smoothing reduced wrinkles but could weaken thin contours.

## 5. Current Optimization Strategy
1. Coarse sampling on clustered ROI representatives.
2. Refine around best candidates.
3. Adaptive rounds with step-size narrowing near top scores.

## 6. Scoring Priorities (High -> Low)
1. Continuity of the outer boundary line
2. Band-fit (outer boundary alignment)
3. Coverage
4. Thickness control
5. Intrusion/outside suppression
6. Wrinkle/endpoints/branch penalties

## 7. Known Limitations
- GPU acceleration not enabled by default (CPU parallel used).
- Score values can be extremely small; GUI supports log or scaled display.
- Large auto-config ranges can still cause long optimization runtime.

## 8. Suggested Next Steps
- Optional GPU backend (CuPy/PyTorch) for scoring and evaluation.
- Model-based optimization (Bayesian / surrogate) for faster convergence.
- More annotated real-world datasets to calibrate scoring targets.

## 9. Testing
- `python3 -m unittest discover`
- `python3 edge_performance_eval.py` for synthetic stress scenarios.

# Formal Development Document (English Only)

## 1. Document Control
- Document Title: OLED FCB Edge Detection System
- Version: 20
- Date: 2026-02-09
- Owner: Edge Detection Team

## 2. Purpose
Define a robust, offline edge detection system that traces the outer boundary
of OLED FCB side-view images for line tracking and quality validation.

## 3. Scope
This document covers the algorithm, GUI workflow, auto-optimization, evaluation
tools, and deployment guidelines.

## 4. Definitions
- **Boundary Band**: Allowed region around the estimated object boundary.
- **Continuity**: Metric for connected, unbroken outer edges.
- **Intrusion**: Percentage of edges inside the object interior.
- **Wrinkle**: Jagged or wavy contour artifacts.
- **Endpoints**: Unwanted edge breaks that indicate discontinuity.

## 5. System Overview
Input images are pre-processed (median/blur/contrast). A Sobel pipeline with NMS,
double-thresholding, and hysteresis extracts edges. Post-filters (polarity and
boundary band) reduce inward curling. Optional smoothing and spur pruning reduce
wrinkle artifacts.

Auto-optimization performs candidate generation, evaluation, and adaptive
re-search with priority-based scoring.

## 6. Functional Requirements
1. Batch processing up to 100 images.
2. Export edge overlay images and coordinate text files.
3. ROI-based optimization with cache support.
4. Auto-optimization progress and graphs.
5. Save/load parameters and auto configuration.

## 7. Architecture
GUI -> Parameter settings -> Edge pipeline -> Output  
Auto optimization -> Candidate generation -> ROI/cluster evaluation -> Apply best

## 8. Evaluation and Validation
- Synthetic bending-loop and complex-loop test images.
- Low-quality, blur, noise, and gradient stress tests.
- Metrics: continuity, band-fit, thickness, intrusion, wrinkle, endpoints.

## 9. Development Process Summary
1. Baseline Sobel implementation and vectorization for performance.
2. NMS relax and thinning to improve recall (with thickness tradeoffs).
3. Polarity and boundary band filters to reduce inner intrusion.
4. Auto thresholding and contrast controls for low-quality images.
5. Multi-stage auto-optimization with adaptive sampling.
6. ROI clustering to reduce evaluation time on large datasets.

## 10. Risks
1. Parameter sensitivity across datasets.
2. Over-smoothing of faint or thin boundaries.
3. Large search ranges can increase optimization time.

## 11. Maintenance Guidance
- Narrow auto config ranges for each production dataset.
- Separate ROI cache per machine or dataset.
- Re-run evaluation after major parameter changes.

## 12. Change Log
- v13: Expanded scoring (continuity/band fit)
- v14: ROI cache + multi-graph GUI + ETA
- v15: Adaptive step search, wrinkle/endpoints penalties, zoomable graphs
- v17: Expanded parameter space and score display scaling
- v18: Tested in production; score display and graph UX improvements
- v19: Score display ×10¹⁵ (display only; learning unchanged), log/scaled modes; x1e9 removed; graph styling (spacing, thin lines, professional theme); full-window scroll margin; documentation update
- v20: **Perfect** auto mode (importance-weighted 2–10× denser grid, ~5× time, coordinate descent); **raw** score display in exponential notation; Wiki-driven updates (score balance, global best, 500 images, score in report/GUI)
- v20 (score & optimization): **Score** — Priority weights: continuity ≥20× base (default 24), band_fit ≥10× (default 12), thickness 1.2, others 1.0; **weighted geometric mean** (log-space) so 끊어짐(discontinuity) and poor band fit strongly penalize; sigmoid clipped for stability. **Auto optimization** — Refine/adaptive **early exit** when no improvement (35% / 40% of candidates); **exploitation-heavy** adaptive (step_scale 0.45→0.12, more centers around best); fewer redundant evals, faster convergence.

## Appendix A: Operator Quick Guide
1. Select images -> set ROI -> run Auto Optimize.
2. Click graphs to zoom/drag and inspect trends.
3. Apply best settings and run batch processing.

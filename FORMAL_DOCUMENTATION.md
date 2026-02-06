# Formal Development Document (English Only)

## 1. Document Control
- Document Title: OLED FCB Edge Detection System
- Version: 17
- Date: 2026-02-04
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

## Appendix A: Operator Quick Guide
1. Select images -> set ROI -> run Auto Optimize.
2. Click graphs to zoom/drag and inspect trends.
3. Apply best settings and run batch processing.

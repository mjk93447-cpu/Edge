# Development Notes (Final Release Readiness)

## Overview
This document summarizes the development and stabilization work for the OLED FCB
edge detection system, including GUI, auto-optimization, evaluation tooling, and
deployment readiness. It is intended to help future maintainers quickly
understand the major decisions and current state.

## Timeline Highlights
- Initial Sobel-based edge detector implemented with vectorized NumPy kernels.
- GUI added for batch processing (up to 100 images) and output export.
- Auto-optimization introduced (coarse → refine → adaptive search).
- Robustness improved for low quality and ambiguous boundary images.
- ROI editor added with caching for repeated datasets.
- Real-time optimization graphs and progress logging added.
- Auto-optimization scoring expanded with continuity/band-fit priorities.

## Key Decisions
- **Vectorization over Python loops** for speed and stable runtime.
- **Boundary band filtering** to reduce inner-edge intrusion.
- **Polarity filtering** to remove inward curves.
- **Auto-threshold scaling** to handle low-contrast inputs.
- **Multi-stage search** (coarse/refine/adaptive) for best results.
- **ROI clustering** for faster optimization on large datasets.

## Current Optimization Strategy
1. **Coarse sampling** on clustered ROI representatives.
2. **Refine** around best candidates.
3. **Adaptive rounds** with step-size narrowing near top scores.

## Scoring Priorities (High → Low)
1. Continuity of the outer boundary line
2. Band-fit (outer boundary alignment)
3. Coverage
4. Thickness control
5. Intrusion/outside suppression
6. Wrinkle/endpoints/branch penalties

## Known Limitations
- GPU acceleration not enabled by default (CPU parallel used).
- Score values can be extremely small; GUI uses log or x1e9 display for visibility.
- Auto-optimization can still be long on large datasets if constraints are wide.

## Suggested Next Steps
- Add optional GPU backend (CuPy/PyTorch) for edge evaluation and scoring.
- Add Bayesian or model-based optimization for faster convergence.
- Improve automated test coverage for extreme optical noise cases.

## Testing
- `python3 -m unittest discover`
- `python3 edge_performance_eval.py` for synthetic stress scenarios.

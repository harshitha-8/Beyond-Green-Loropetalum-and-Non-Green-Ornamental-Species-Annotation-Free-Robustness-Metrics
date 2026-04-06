# Camera-Ready Paper Alignment (CVPR 2026 V4A)

This note captures the canonical metadata and key quantitative values used to align repository text with the camera-ready paper.

## Canonical Title

`Beyond Green: Domain-Aware Self-Validated Instance Counting for Loropetalum and Non-Green Ornamental Species using Annotation-Free Robustness Metrics`

## Authors (Order)

1. Harshitha Manjunatha
2. Prabha Sundaravadivel (corresponding)
3. Shekhar Suman Borah
4. Lakshman Tamil
5. Patricia R. Knight
6. H. Allen Torbert
7. Siva P. Kumpatla

## Core Dataset Stats

- Dataset: Loropetalum chinense UAV Dataset
- Images: 469
- Resolution: 4000 x 3000
- Plant count range: 797-1,789
- Mean plants/image: 1,398 +/- 267
- Typical annotation effort: 190-380 min/image

## Deployment Thresholds (Production)

- RCS > 0.95
- CSC > 0.90
- SVS > 0.55
- ARI > 0.85

## Aggregate Robustness (469 Images)

- RCS: 0.981 +/- 0.015 (min 0.958, max 0.998)
- CSC: 0.932 +/- 0.038 (min 0.770, max 0.993)
- SVS: 0.598 +/- 0.024 (min 0.557, max 0.634)
- ARI: 0.919 +/- 0.087 (min 0.685, max 0.994)

## Baseline Counting Model (Test Set)

- MAE: 23.7 +/- 8.2
- RMSE: 31.4
- Relative error: 1.85% (mean count 1,412)

## Efficiency and Deployment Signal

- Validation time reduction: ~15x vs full manual annotation
- CLIP-based stratified sampling speedup: ~35x (within metric pipeline)
- Correlation with deployment success: rho = 0.87, p < 0.001

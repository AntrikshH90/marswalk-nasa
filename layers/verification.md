# MarsWalk Validation Report
Generated: 2026-09-23T14:09:07.498645

## 1. Ground Truth Summary
- **Source**: NASA/NAIF M2020 surface rover location SPK kernels, body -168, frame IAU_MARS, target Mars (499)
- **Total waypoints**: 4178
- **Sol range**: [0, 1980]
- **Total distance**: 44.05 km
- **BBox**: lon 77.2168 to 18.4200, lat 77.4631 to 18.4997

## 2. Validation Split (Time-Based)
- **Training positives**: 1693
- **Training negatives**: 796
- **Notes**: ['Positives: Perseverance waypoints, sol 0-800 (training split)', 'Negatives: Outer ring terrain (physically derived; replace with THEMIS+DTM slope)', 'This file replaces synthetic labels in train_ml.py']

## 3. Model Validation Metrics
- **AUC-ROC**: N/A
- **Accuracy**: N/A
- **Precision**: N/A
- **Recall**: N/A

### Sources

### Validation
- Method: N/A
- Train sols: N/A
- Test sols: N/A

## 4. Uncertainty Analysis
- **Mean uncertainty**: 0.0000
- **Max uncertainty**: 0.0000
- **95th percentile**: 0.0000
- **Cells with high uncertainty (>0.2)**: 0

## 5. Recommendations for Operational Use
1. **Use uncertainty threshold**: Flag cells with uncertainty > 0.2 as requiring human review
2. **Update with new drives**: Retrain on sols 0-900 when new drive data becomes available
3. **Verify with THEMIS**: Load thermal inertia/rock abundance for hazard assessment
4. **Calibrate per-sol**: Solar longitude (LS) affects dust; add seasonal features for long-term operations

## 6. Documented Limitations
- **No negative observations**: Rover avoided terrain for science reasons, not just impassability
- **Single DTM**: Vertical error ±0.5m; uncertainty analysis accounts for this
- **Local coverage**: Only Jezero Crater; requires new DTM for other sites
- **No wheel-slip data**: Traversability = binary (go/no-go), not continuous slip metric

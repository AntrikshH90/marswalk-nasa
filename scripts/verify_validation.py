#!/usr/bin/env python3
"""
Verification and validation report generator.

Outputs a markdown report with:
- Ground truth data summary
- Model validation metrics
- Uncertainty analysis
- Recommendations for operational use
"""
import json
import os
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
LAYERS = os.path.join(ROOT, "layers")
MODELS = os.path.join(ROOT, "models")


def load_report():
    """Generate validation report."""
    lines = []
    lines.append("# MarsWalk Validation Report")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    
    # 1. Ground Truth Summary
    lines.append("## 1. Ground Truth Summary")
    traverse_path = os.path.join(DATA_DIR, "perseverance_traverse.json")
    with open(traverse_path, 'r') as f:
        traverse = json.load(f)
    
    lines.append(f"- **Source**: {traverse['source']}")
    lines.append(f"- **Total waypoints**: {traverse['n_samples']}")
    lines.append(f"- **Sol range**: {traverse['sol_range']}")
    lines.append(f"- **Total distance**: {traverse['distance_km']:.2f} km")
    lines.append(f"- **BBox**: lon {traverse['bbox_lonlat'][0]:.4f} to {traverse['bbox_lonlat'][1]:.4f}, lat {traverse['bbox_lonlat'][2]:.4f} to {traverse['bbox_lonlat'][3]:.4f}")
    lines.append("")
    
    # 2. Train/Test Split
    lines.append("## 2. Validation Split (Time-Based)")
    train_path = os.path.join(LAYERS, "train_labels.json")
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            labels = json.load(f)
        lines.append(f"- **Training positives**: {len(labels['positives'])}")
        lines.append(f"- **Training negatives**: {len(labels['negatives'])}")
        lines.append(f"- **Notes**: {labels.get('notes', [])}")
    lines.append("")
    
    # 3. Model Metrics
    lines.append("## 3. Model Validation Metrics")
    metrics_path = os.path.join(MODELS, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        m = metrics.get('metrics', {})
        lines.append(f"- **AUC-ROC**: {m.get('auc', 'N/A')}")
        lines.append(f"- **Accuracy**: {m.get('accuracy', 'N/A')}")
        lines.append(f"- **Precision**: {m.get('precision', 'N/A')}")
        lines.append(f"- **Recall**: {m.get('recall', 'N/A')}")
        lines.append("")
        lines.append("### Sources")
        for src, url in metrics.get('sources', {}).items():
            lines.append(f"- {src}: {url}")
        lines.append("")
        lines.append("### Validation")
        val = metrics.get('validation', {})
        lines.append(f"- Method: {val.get('method', 'N/A')}")
        lines.append(f"- Train sols: {val.get('train_sols', 'N/A')}")
        lines.append(f"- Test sols: {val.get('test_sols', 'N/A')}")
    else:
        lines.append("- No metrics found. Run `train_traversability.py` first.")
    lines.append("")
    
    # 4. Uncertainty Analysis
    lines.append("## 4. Uncertainty Analysis")
    uncertainty_path = os.path.join(LAYERS, "uncertainty.npy")
    if os.path.exists(uncertainty_path):
        unc = np.load(uncertainty_path)
        lines.append(f"- **Mean uncertainty**: {unc.mean():.4f}")
        lines.append(f"- **Max uncertainty**: {unc.max():.4f}")
        lines.append(f"- **95th percentile**: {np.percentile(unc, 95):.4f}")
        lines.append(f"- **Cells with high uncertainty (>0.2)**: {(unc > 0.2).sum()}")
    else:
        lines.append("- No uncertainty map found. Run `train_traversability.py` first.")
    lines.append("")
    
    # 5. Recommendations
    lines.append("## 5. Recommendations for Operational Use")
    lines.append("1. **Use uncertainty threshold**: Flag cells with uncertainty > 0.2 as requiring human review")
    lines.append("2. **Update with new drives**: Retrain on sols 0-900 when new drive data becomes available")
    lines.append("3. **Verify with THEMIS**: Load thermal inertia/rock abundance for hazard assessment")
    lines.append("4. **Calibrate per-sol**: Solar longitude (LS) affects dust; add seasonal features for long-term operations")
    lines.append("")
    
    # 6. Limitations
    lines.append("## 6. Documented Limitations")
    lines.append("- **No negative observations**: Rover avoided terrain for science reasons, not just impassability")
    lines.append("- **Single DTM**: Vertical error ±0.5m; uncertainty analysis accounts for this")
    lines.append("- **Local coverage**: Only Jezero Crater; requires new DTM for other sites")
    lines.append("- **No wheel-slip data**: Traversability = binary (go/no-go), not continuous slip metric")
    lines.append("")
    
    return "\n".join(lines)


def save_report():
    """Save report to verification.md."""
    report = load_report()
    output = os.path.join(LAYERS, "verification.md")
    with open(output, 'w') as f:
        f.write(report)
    print(f"Saved verification report to {output}")


def main():
    report = load_report()
    print(report)
    print()
    save_report()


if __name__ == "__main__":
    main()

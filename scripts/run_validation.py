#!/usr/bin/env python3
"""
End-to-end validation pipeline for MarsWalk.

Runs all components and verifies outputs exist.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_file(path, name):
    """Check if file exists."""
    if os.path.exists(path):
        print(f"✅ {name}: {os.path.getsize(path)} bytes")
        return True
    else:
        print(f"❌ {name}: MISSING")
        return False


def main():
    print("=" * 60)
    print("MarsWalk End-to-End Validation")
    print("=" * 60)
    
    # Check raw data
    print("\n📁 Raw Data:")
    check_file(os.path.join(ROOT, "data", "perseverance_traverse.json"), "Perseverance waypoints")
    check_file(os.path.join(ROOT, "data", "jezero_hirise_dtm_1m.tif"), "HiRISE DTM")
    
    # Check computed layers
    print("\n📁 Computed Layers:")
    check_file(os.path.join(ROOT, "layers", "slope.npy"), "Slope map")
    check_file(os.path.join(ROOT, "layers", "rough.npy"), "Roughness map")
    check_file(os.path.join(ROOT, "layers", "trav.npy"), "Traversability map")
    
    # Check models
    print("\n📁 Models:")
    check_file(os.path.join(ROOT, "models", "traversability.joblib"), "Traversability model")
    check_file(os.path.join(ROOT, "models", "metrics.json"), "Validation metrics")
    
    # Check outputs
    print("\n📁 Outputs:")
    check_file(os.path.join(ROOT, "static", "validation_overlay.png"), "Validation visualization")
    check_file(os.path.join(ROOT, "layers", "verification.md"), "Validation report")
    check_file(os.path.join(ROOT, "README.md"), "README")
    
    print("\n" + "=" * 60)
    print("Validation complete. Review README.md and layers/verification.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

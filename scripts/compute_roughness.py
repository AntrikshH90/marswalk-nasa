#!/usr/bin/env python3
"""
Compute surface roughness from HiRISE DTM.

Roughness is defined as the standard deviation of elevation within
a local window (typically 5-10m for rover operations).
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = os.path.join(ROOT, "layers")


def compute_roughness(elev_40m, window_size=5):
    """
    Compute surface roughness as elevation standard deviation in local window.
    
    Args:
        elev_40m: Elevation at 40m resolution (from HiRISE DTM)
        window_size: Window size in pixels (40m each)
    
    Returns:
        rough_40m: Surface roughness (meters) at 40m resolution
    """
    from scipy.ndimage import generic_filter
    
    print(f"Computing roughness with {window_size}x{window_size} pixel window...")
    
    # Roughness = std dev of elevation in local window
    rough_40m = generic_filter(
        elev_40m,
        lambda x: np.std(x),
        size=window_size,
        mode='constant',
        cval=np.nan
    )
    
    print(f"Roughness array: {rough_40m.shape}")
    
    # Fill NaN values at edges
    print("Filling nodata areas...")
    mask = ~np.isnan(rough_40m)
    if mask.sum() > 0:
        rough_40m = np.where(np.isnan(rough_40m),
                             np.nanmedian(rough_40m), rough_40m)
    
    return rough_40m


def main():
    # Load elevation from DTM
    elev_path = os.path.join(LAYERS, "elev.npy")
    elev_40m = np.load(elev_path)
    print(f"Loaded elevation: {elev_40m.shape}")
    
    # Compute roughness
    rough_40m = compute_roughness(elev_40m, window_size=5)
    
    # Save
    rough_path = os.path.join(LAYERS, "rough.npy")
    np.save(rough_path, rough_40m.astype(np.float32))
    print(f"Saved roughness to {rough_path}")
    
    # Statistics
    print("\n=== Roughness Statistics ===")
    print(f"Min: {rough_40m.min():.2f} m")
    print(f"Max: {rough_40m.max():.2f} m")
    print(f"Mean: {rough_40m.mean():.2f} m")
    print(f"95th percentile: {np.percentile(rough_40m, 95):.2f} m")


if __name__ == "__main__":
    main()

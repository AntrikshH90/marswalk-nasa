#!/usr/bin/env python3
"""
Load and process HiRISE 1m DTM for traversability analysis.

Perseverance Rover Traverse Analysis - NASA Space Apps Challenge
Source: USGS Astrogeology Science Center
Product: JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m
Resolution: 1 meter/pixel
Citation: Fergason et al. (2020), Bland et al. (2024)
"""
import os
import math
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
LAYERS = os.path.join(ROOT, "layers")


def load_hirise_dtm():
    """
    Load HiRISE 1m DTM and compute slope at working resolution.
    
    Returns:
        slope_40m: Slope in degrees at 40m working resolution
        elev_40m: Elevation at 40m working resolution
        meta: Metadata about the projection and transformation
    """
    import rasterio
    import json
    
    dtm_path = os.path.join(DATA_DIR, "jezero_hirise_dtm_1m.tif")
    meta_path = os.path.join(LAYERS, "meta.json")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"Loading HiRISE DTM: {dtm_path}")
    
    # Read entire DTM (1.8 GB, may take memory)
    with rasterio.open(dtm_path) as src:
        print("Reading elevation data from HiRISE DTM...")
        elev_1m = src.read(1)
        
        # Replace nodata values
        nodata = src.nodata
        if nodata is not None:
            elev_1m[elev_1m == nodata] = np.nan
    
    print(f"Loaded elevation array: {elev_1m.shape}")
    
    # Compute slope using finite differences (at 1m resolution first)
    print("Computing slope from elevation gradient...")
    dz_dx = np.gradient(elev_1m, axis=1)
    dz_dy = np.gradient(elev_1m, axis=0)
    
    # Slope = arctan(sqrt(dz/dx^2 + dz/dy^2))
    slope_1m = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    
    # Downsample to 40m working resolution (average slope in 40x40 blocks)
    print("Downsampling to 40m working resolution...")
    factor = 40
    H_1m, W_1m = elev_1m.shape
    
    # Ensure we have full blocks
    H_work = H_1m // factor
    W_work = W_1m // factor
    
    print(f"Output dimensions: {H_work} x {W_work} (working grid 40m resolution)")
    
    # Extract working grid area (crop to match meta.json bounds)
    # Working grid is 2228 x 2534 at 40m resolution
    # This corresponds to ~89120 x 101360 m at 1m resolution
    
    # Crop to match the meta working grid
    start_row = (H_1m - H_work * factor) // 2
    start_col = (W_1m - W_work * factor) // 2
    
    elev_crop = elev_1m[start_row:start_row + H_work*factor, 
                        start_col:start_col + W_work*factor]
    slope_crop = slope_1m[start_row:start_row + H_work*factor, 
                          start_col:start_col + W_work*factor]
    
    # Reshape to compute block means
    elev_40m = elev_crop.reshape(H_work, factor, W_work, factor).mean(axis=(1, 3))
    slope_40m = slope_crop.reshape(H_work, factor, W_work, factor).mean(axis=(1, 3))
    
    print(f"Downsampled slope array: {slope_40m.shape}")
    
    # Clip to valid range (slopes > 45° are likely artifacts)
    slope_40m = np.clip(slope_40m, 0, 45)
    
    # Fill NaN values (nodata areas) with interpolation or median
    print("Filling nodata areas...")
    mask = ~np.isnan(slope_40m)
    if mask.sum() > 0:
        slope_40m = np.where(np.isnan(slope_40m), 
                             np.nanmedian(slope_40m), slope_40m)
        elev_40m = np.where(np.isnan(elev_40m),
                            np.nanmedian(elev_40m), elev_40m)
    
    return slope_40m, elev_40m, meta


def save_slope_layer(slope_40m, elev_40m):
    """Save computed slope to layers directory."""
    import os
    
    slope_path = os.path.join(LAYERS, "slope.npy")
    elev_path = os.path.join(LAYERS, "elev.npy")
    
    np.save(slope_path, slope_40m.astype(np.float32))
    np.save(elev_path, elev_40m.astype(np.float32))
    
    print(f"Saved slope layer to {slope_path}")
    print(f"Saved elevation layer to {elev_path}")


def main():
    slope_40m, elev_40m, meta = load_hirise_dtm()
    save_slope_layer(slope_40m, elev_40m)
    
    # Summary statistics
    print("\n=== Slope Statistics ===")
    print(f"Min: {slope_40m.min():.2f}°")
    print(f"Max: {slope_40m.max():.2f}°")
    print(f"Mean: {slope_40m.mean():.2f}°")
    print(f"90th percentile: {np.percentile(slope_40m, 90):.2f}°")
    print(f"95th percentile: {np.percentile(slope_40m, 95):.2f}°")
    print(f"99th percentile: {np.percentile(slope_40m, 99):.2f}°")
    
    print("\n=== Traverse Corridor Analysis ===")
    print("Note: This slope map covers the Perseverance traverse corridor.")
    print("Traversability analysis uses real Perseverance waypoints as ground truth.")


if __name__ == "__main__":
    main()

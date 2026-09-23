#!/usr/bin/env python3
"""
Visualize Perseverance track against HiRISE DTM-derived terrain.

Produces publication-quality figures for validation.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
LAYERS = os.path.join(ROOT, "layers")
STATIC = os.path.join(ROOT, "static")


def load_track():
    """Load Perseverance track data."""
    traverse_path = os.path.join(DATA_DIR, "perseverance_traverse.json")
    with open(traverse_path, 'r') as f:
        traverse = json.load(f)
    return traverse["track"], traverse


def load_layers():
    """Load precomputed terrain layers."""
    slope = np.load(os.path.join(LAYERS, "slope.npy"))
    rough = np.load(os.path.join(LAYERS, "rough.npy"))
    trav = np.load(os.path.join(LAYERS, "trav.npy"))
    meta_path = os.path.join(LAYERS, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return slope, rough, trav, meta


def plot_traverse_on_terrain(track, slope, trav, meta):
    """Plot track overlaid on slope and traversability maps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Compute track coordinates
    bbox = meta["bbox"]
    H, W = slope.shape
    rows, cols = [], []
    
    for pt in tqdm(track, desc="Projecting track"):
        lon, lat = pt["lon"], pt["lat"]
        col = (lon - bbox["w"]) / (bbox["e"] - bbox["w"]) * W
        row = (bbox["n"] - lat) / (bbox["n"] - bbox["s"]) * H
        rows.append(row)
        cols.append(col)
    
    rows, cols = np.array(rows), np.array(cols)
    sol = np.array([pt["sol"] for pt in track])
    
    # Plot 1: Slope Map
    ax = axes[0, 0]
    im = ax.imshow(slope, cmap='terrain', origin='upper', 
                   extent=[bbox['w'], bbox['e'], bbox['s'], bbox['n']])
    ax.plot(cols, rows, 'r-', linewidth=0.5, alpha=0.8)
    ax.set_title(f"HiRISE DTM Slope (n={len(track)})")
    fig.colorbar(im, ax=ax, label="Slope (°)")
    
    # Plot 2: Traversability Map
    ax = axes[0, 1]
    im = ax.imshow(trav, cmap='coolwarm', origin='upper',
                   extent=[bbox['w'], bbox['e'], bbox['s'], bbox['n']])
    ax.plot(cols, rows, 'r-', linewidth=0.5, alpha=0.8)
    ax.set_title("Traversability Probability (Model Output)")
    fig.colorbar(im, ax=ax, label="P(traversable)")
    
    # Plot 3: Sol Progress
    ax = axes[1, 0]
    ax.plot(sol, rows, 'b-', alpha=0.6)
    ax.set_xlabel("Sol")
    ax.set_ylabel("Row (North-South)")
    ax.set_title("Sol Progression")
    ax.invert_yaxis()
    
    # Plot 4: Slope Along Track
    ax = axes[1, 1]
    track_slope = slope[np.minimum(rows, H-1).astype(int), 
                        np.minimum(cols, W-1).astype(int)]
    ax.plot(sol, track_slope, 'g-', alpha=0.6)
    ax.axhline(y=20, color='r', linestyle='--', label='20° limit')
    ax.set_xlabel("Sol")
    ax.set_ylabel("Slope (°)")
    ax.set_title("Slope Along Traverse Path")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC, "validation_overlay.png"), dpi=150, bbox_inches='tight')
    print(f"Saved validation overlay to {os.path.join(STATIC, 'validation_overlay.png')}")
    plt.close()


def main():
    print("Loading data...")
    track, traverse = load_track()
    slope, rough, trav, meta = load_layers()
    
    print(f"Loaded {len(track)} waypoints over sols {traverse['sol_range']}")
    print(f"Terrain layers: {slope.shape}")
    
    print("Generating validation visualization...")
    plot_traverse_on_terrain(track, slope, trav, meta)
    
    print("\nVisualization complete. Check static/validation_overlay.png")


if __name__ == "__main__":
    main()

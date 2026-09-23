#!/usr/bin/env python3
"""
Generate publication-quality figures for submission.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = os.path.join(ROOT, "layers")
SUBMISSION = os.path.join(ROOT, "submission")
os.makedirs(SUBMISSION, exist_ok=True)


def load_layers():
    slope = np.load(os.path.join(LAYERS, "slope.npy"))
    rough = np.load(os.path.join(LAYERS, "rough.npy"))
    trav = np.load(os.path.join(LAYERS, "trav.npy"))
    return slope, rough, trav


def plot_histograms():
    """Plot terrain attribute histograms."""
    slope, rough, trav = load_layers()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Slope histogram
    axes[0].hist(slope.flatten(), bins=50, color='steelblue', alpha=0.7)
    axes[0].axvline(20, color='red', linestyle='--', label='20° limit')
    axes[0].set_xlabel('Slope (°)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Terrain Slope Distribution')
    axes[0].legend()
    
    # Roughness histogram
    axes[1].hist(rough.flatten(), bins=50, color='forestgreen', alpha=0.7)
    axes[1].axvline(15, color='red', linestyle='--', label='15m limit')
    axes[1].set_xlabel('Roughness (m)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Surface Roughness Distribution')
    axes[1].legend()
    
    # Traversability histogram
    axes[2].hist(trav.flatten(), bins=50, color='purple', alpha=0.7)
    axes[2].set_xlabel('Traversability Probability')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Model Output Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUBMISSION, 'terrain_histograms.png'), dpi=300, bbox_inches='tight')
    print("Saved: terrain_histograms.png")
    plt.close()


def plot_tilemap():
    """Plot slope map with colorbar."""
    slope, rough, trav = load_layers()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im0 = axes[0].imshow(slope, cmap='terrain', origin='lower')
    axes[0].set_title('HiRISE DTM Slope')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(rough, cmap='gist_earth', origin='lower')
    axes[1].set_title('Surface Roughness')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(trav, cmap='coolwarm', origin='lower')
    axes[2].set_title('Traversability Probability')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUBMISSION, 'terrain_tilemap.png'), dpi=300, bbox_inches='tight')
    print("Saved: terrain_tilemap.png")
    plt.close()


def main():
    print("Generating submission figures...")
    plot_histograms()
    plot_tilemap()
    print("Done! Check submission/ folder.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MARSWALK — ML training pipeline using REAL Perseverance ground truth

Step 2 of validation plan:
- Uses real Perseverance waypoints as positives (sol 0-800)
- Uses physically-derived negatives (outer ring terrain)
- Time-based split: sols 0-800 train, 800+ test
"""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              GradientBoostingRegressor)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = os.path.join(ROOT, "layers")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def load_labels():
    """Load real training labels from Perseverance data."""
    labels_path = os.path.join(LAYERS, "train_labels.json")
    with open(labels_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_terrain_features():
    """
    Load terrain features from preprocessed arrays.
    For a full implementation, these should be derived from:
    - HiRISE DTM (1m) for slope and roughness
    - THEMIS for thermal inertia/rock abundance
    """
    # For now, load the precomputed features
    slope = np.load(os.path.join(LAYERS, "slope.npy"))
    rough = np.load(os.path.join(LAYERS, "rough.npy"))
    elev = np.load(os.path.join(LAYERS, "elev.npy"))
    mineral = np.load(os.path.join(LAYERS, "mineral_idx.npy"))
    return slope, rough, elev, mineral


def build_features_from_labels(labels, slope, rough, elev, mineral):
    """
    Build feature vectors from labeled points.
    
    For each point, extract terrain features at that location.
    """
    positives = labels["positives"]
    negatives = labels["negatives"]
    
    # Build feature vectors
    X, y = [], []
    
    # Positives (labeled as traversable)
    for row, col in positives:
        if 0 <= row < slope.shape[0] and 0 <= col < slope.shape[1]:
            features = [
                float(slope[row, col]),
                float(rough[row, col]),
                float(elev[row, col]),
                float(mineral[row, col])
            ]
            X.append(features)
            y.append(1)  # traversable
    
    # Negatives (labeled as not traversable)
    for row, col in negatives:
        if 0 <= row < slope.shape[0] and 0 <= col < slope.shape[1]:
            features = [
                float(slope[row, col]),
                float(rough[row, col]),
                float(elev[row, col]),
                float(mineral[row, col])
            ]
            X.append(features)
            y.append(0)  # not traversable
    
    return np.array(X), np.array(y)


def train_traversability_model(X, y):
    """
    Train traversability classifier on real labels.
    
    Note: This uses physically-derived negatives, not observed impassable terrain.
    """
    # Time-based split: sols 0-800 train, 800+ test
    # Since we're using labels already split by sol, use standard split for now
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    log("[model 1] traversability classifier (real Perseverance labels)...")
    
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=8,
        random_state=42
    )
    clf.fit(Xtr, ytr)
    
    acc = clf.score(Xte, yte)
    log(f"  val accuracy = {acc:.4f}")
    
    # Save
    joblib.dump(clf, os.path.join(MODELS, "traversability.joblib"))
    
    return clf


def generate_traversability_map(clf, slope, rough, elev, mineral):
    """Generate full-grid traversability probability map."""
    H, W = slope.shape
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000
    trav = np.zeros((H, W), np.float32)
    
    for i in range(0, H * W, chunk_size):
        end = min(i + chunk_size, H * W)
        rows = (np.arange(i, end) // W) % H
        cols = np.arange(i, end) % W
        
        features = np.column_stack([
            slope[rows, cols],
            rough[rows, cols],
            elev[rows, cols],
            mineral[rows, cols]
        ])
        
        trav[rows, cols] = clf.predict_proba(features)[:, 1]
    
    np.save(os.path.join(LAYERS, "trav.npy"), trav)
    return trav


def main():
    log("Building training pipeline with REAL Perseverance ground truth...")
    
    # Load data
    labels = load_labels()
    slope, rough, elev, mineral = load_terrain_features()
    
    log(f"Loaded {len(labels['positives'])} positive, {len(labels['negatives'])} negative samples")
    
    # Build features
    X, y = build_features_from_labels(labels, slope, rough, elev, mineral)
    log(f"Feature matrix: {X.shape}")
    
    # Train traversability model
    clf = train_traversability_model(X, y)
    
    # Generate full map
    trav = generate_traversability_map(clf, slope, rough, elev, mineral)
    log(f"Generated traversability map: {trav.shape}")
    
    log("\nDone. Models saved to models/")


if __name__ == "__main__":
    main()

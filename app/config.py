"""Runtime configuration.

The NASA API key is read from the environment first so it does not have to be
committed. The fallback below is the key this project was developed against.

Getting a key is free at https://api.nasa.gov -- the default DEMO_KEY is
rate-limited to roughly 30 requests/hour, which is not enough for a live demo,
so a real key is required.

NOTE for the team: this file is checked in. If the repository is ever made
public, rotate the key at api.nasa.gov and set NASA_API_KEY in the environment
(or in a .env file that is git-ignored) instead of editing the value here.
"""
import os

NASA_API_KEY = os.environ.get("NASA_API_KEY", "").strip() or "2IuZwjK5tGXzaDsFX2FB3y0PKmJECvSOla9VYtaK"

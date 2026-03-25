#!/usr/bin/env python3
"""Test if .env is being loaded correctly by the medical agent."""

import os
import sys
from pathlib import Path

# Simulate the same path resolution as the medical agent
current_file = Path(__file__).resolve()
print(f"Current file: {current_file}")

# For medical agent: __main__.py -> app -> medical_Images -> agents -> python -> samples -> root
# That's 6 levels up from __main__.py
medical_main = Path("samples/python/agents/medical_Images/app/__main__.py").resolve()
print(f"\nMedical agent __main__.py: {medical_main}")

root_dir = medical_main.parents[5]
print(f"Root dir (parents[5]): {root_dir}")

env_path = root_dir / '.env'
print(f"Env path: {env_path}")
print(f"Env exists: {env_path.exists()}")

# Now load it
from dotenv import load_dotenv
result = load_dotenv(dotenv_path=env_path)
print(f"\nload_dotenv result: {result}")

# Check if variables are loaded
print("\n" + "="*70)
print("Environment Variables:")
print("="*70)
print(f"GOOGLE_API_KEY: {'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Not set'}")
print(f"TAVILY_API_KEY: {'✅ Set' if os.getenv('TAVILY_API_KEY') else '❌ Not set'}")
print(f"QDRANT_URL: {'✅ Set' if os.getenv('QDRANT_URL') else '❌ Not set'}")
print(f"QDRANT_KEY: {'✅ Set' if os.getenv('QDRANT_KEY') else '❌ Not set'}")

if os.getenv('TAVILY_API_KEY'):
    print(f"\nTAVILY_API_KEY value: {os.getenv('TAVILY_API_KEY')[:10]}...")

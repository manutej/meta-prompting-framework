#!/usr/bin/env python3
"""
Run experiments that don't require API key.

Shows what v2 can do now (Phase 1) without needing Anthropic API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("WHAT YOU CAN TEST NOW (No API Key Needed)")
print("="*80)
print()

print("1. v2 Categorical Structure Tests")
print("   → Tests RMP monad, enriched categories, polynomial functors")
print("   → All categorical laws verification")
print()

print("2. Version Information")
print("   → Check which frameworks are available")
print("   → See version compatibility")
print()

print("="*80)
print()

# Test 1: v2 Structure
print("Running Test 1: v2 Categorical Structure...")
print("-"*80)
import subprocess
result = subprocess.run(
    ["python", "experiments/test_v2_structure.py"],
    capture_output=False
)
print("-"*80)
print()

# Test 2: Version Info
print("Running Test 2: Version Information...")
print("-"*80)
result = subprocess.run(
    ["python", "meta_prompting.py"],
    capture_output=False
)
print("-"*80)
print()

print("="*80)
print("WHAT NEEDS API KEY")
print("="*80)
print()
print("To run v1 experiments and real v1 vs v2 comparison:")
print()
print("1. Get Anthropic API key from: https://console.anthropic.com")
print("2. Set environment variable:")
print("   export ANTHROPIC_API_KEY='your-key-here'")
print()
print("3. Then run:")
print("   python experiments/quick_demo.py          # 3 experiments")
print("   python -m experiments.run_suite           # All 10 experiments")
print()
print("Or run existing v1 demo:")
print("   python demo_meta_prompting.py")
print()
print("="*80)

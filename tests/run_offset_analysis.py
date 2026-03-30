#!/usr/bin/env python3
"""
Simple runner script for offset distribution analysis.

This script runs the offset analysis with the correct configuration path.
"""

import os
import sys

# Add the project src directory to Python path
project_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if project_src_root not in sys.path:
    sys.path.insert(0, project_src_root)

# Import and run the analysis
from oft.transformer.scripts.analyze_offset_distributions import main

if __name__ == '__main__':
    main() 
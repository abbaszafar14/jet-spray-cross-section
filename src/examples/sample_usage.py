"""
Example usage of spray cross-section analysis

Before running:
  1. Place your cut-section Mie scattering images in: data/images/
  2. Update INPUT_DIR and OUTPUT_DIR in spray_cross_section_analysis.py
  3. Set pixel calibration (PIXEL_TO_MM) and nozzle origin coordinates
  
Configuration parameters to adjust:
  - INPUT_DIR: Path to cropped Mie scattering images
  - OUTPUT_DIR: Path where results will be saved
  - PIXEL_TO_MM: Pixel calibration factor [mm/pixel]
  - X_ORIGIN_FROM_LEFT_MM: Nozzle x-position from left edge
  - Y_ORIGIN_FROM_TOP_MM: Nozzle y-position from top edge
  - SAVE_FORMAT: "png" or "pdf" for vector graphics
  - DPI: Resolution for saved figures

Then run:
  python src/spray_cross_section_analysis.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import and run the analysis
from spray_cross_section_analysis import *

print("Cross-section analysis complete!")
print("Check the results directory for plots (PNG/PDF), CSV, and NPZ files.")

# Cartesian-Based Automatic Ironing Robot
### Vision-Guided Steam and Vacuum End-Effector

> **Published:** Design and Implementation of a Cartesian-Based Automatic Ironing Robot Using Vision-Guided Steam and Vacuum End-Effector — *ICAIT 2025 (Scopus-indexed)*

---

## Overview

A fully functional autonomous ironing robot built on a Cartesian (3-axis) frame. The robot uses a camera mounted above the ironing surface to detect a garment, generate a coverage path, and drive the end-effector — which combines a steam iron and vacuum suction — across the entire cloth surface automatically.

This repository contains the complete vision pipeline: from raw image capture through garment segmentation, contour detection, coordinate conversion, and ironing path generation.

---
## Demo

https://github.com/Jeswin010/cartesian-ironing-robot/blob/main/robot_demo.mp4
The demo video shows an early hardware test. The current codebase reflects the complete 
pipeline.

## System Architecture

```
Camera Input
     │
     ▼
Undistortion (calibrated intrinsics)
     │
     ▼
GrabCut Segmentation  ──►  Garment Mask
     │
     ▼
Canny Edge Detection + Contour Extraction
     │
     ▼
Douglas-Peucker Simplification
     │
     ▼
Pinhole Model → Real-World Coordinates (mm)
     │
     ▼
Boustrophedon Path Generation
     │
     ▼
Waypoints → Arduino (Serial) → Stepper Motors
```

---

## Hardware

| Component | Details |
|---|---|
| Frame | Cartesian XY gantry (custom-built) |
| Motors | NEMA 17 stepper motors |
| Driver | DRV8825 on CNC Shield (Arduino Uno) |
| End-effector | Steam iron head + vacuum suction cup |
| Camera | USB webcam (calibrated, mounted overhead) |
| Controller | Arduino Uno + Python (serial communication) |

---

## Vision Pipeline

### 1. Garment Segmentation — GrabCut
Rather than simple colour thresholding, GrabCut models the colour distribution of both foreground (garment) and background using Gaussian Mixture Models, then finds the optimal boundary via graph-cut. This handles white shirts on white backgrounds where HSV thresholding fails.

### 2. Contour Detection — Canny + Filters
Canny edge detection is applied on the segmentation mask (not the raw image), isolating only the garment boundary. Contours are filtered by:
- Area > 10% of image
- Width and height > 20% of image dimensions
- Aspect ratio between 0.3 and 3.0 (rejects hanger rods automatically)

### 3. Boundary Simplification — Douglas-Peucker
`cv2.approxPolyDP` reduces contour point density while preserving boundary shape, as described in the paper.

### 4. Coordinate Conversion — Pinhole Model
```
X_mm = (u - cx) × Z / fx
Y_mm = (v - cy) × Z / fy
```
Camera height Z = 750 mm. Calibrated intrinsics: fx=569.75, fy=568.89, cx=339.38, cy=215.55.

### 5. Path Generation — Boustrophedon
- Start: bottom-right of garment mask
- Even columns: reposition upward (ironing OFF)
- Odd columns: iron downward (steam + vacuum ON)
- Strip spacing derived from physical end-effector width (30 mm)

---

## Repository Structure

```
cartesian-ironing-robot/
├── ironing_pipeline.py      # Main vision + path planning pipeline
├── calibration_data.npz     # Camera calibration matrices
├── paper/
│   └── ICAIT2025_paper.pdf  # Published conference paper
├── media/
│   ├── robot_setup.jpg      # Physical robot photograph
│   └── simulation.gif       # End-effector path animation
└── README.md
```

---

## Requirements

```
Python >= 3.8
opencv-python
numpy
matplotlib
pyserial          # for Arduino communication (next iteration)
```

Install with:
```bash
pip install opencv-python numpy matplotlib pyserial
```

---

## Usage

```bash
# Run with default image path (edit IMAGE_PATH in script)
python ironing_pipeline.py

# Or pass image path as argument
python ironing_pipeline.py --image "path/to/shirt.jpg"
```

**Output:**
- 4-panel static visualisation saved as `pipeline_output.png`
- Animated end-effector simulation (live window)
- Waypoint coordinates printed to terminal

---

## Results

The pipeline successfully:
- Segments garments from plain and near-white backgrounds
- Generates complete coverage paths adapting to garment shape
- Converts pixel coordinates to real-world mm using calibrated camera model
- Produces waypoints ready for serial transmission to Arduino

*Serial communication to Arduino and physical robot integration code to be added in the next commit.*

---

## Paper

If you use or reference this work:

> [Your Name(s)], "Design and Implementation of a Cartesian-Based Automatic Ironing Robot Using Vision-Guided Steam and Vacuum End-Effector", *Proceedings of ICAIT 2025*.

---

## Author

**[Your Full Name]**  
B.Tech — [Your Department], [Your College Name]  
[Your LinkedIn URL]  
[Your Email]

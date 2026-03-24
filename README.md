# Cartesian-Based Automatic Ironing Robot

An autonomous ironing robot that detects cloth boundaries using computer vision 
and irons garments using a steam-vacuum end effector mounted on a Cartesian platform.

## System Overview
- Belt-driven Cartesian frame (1000×800 mm), aluminium extrusions, V-slot wheels
- NEMA17 stepper motors with DRV8825 drivers (1/16th microstepping)
- Arduino Uno + CNC Shield for motion control
- Webcam at fixed 75 cm distance for cloth detection

## Vision Pipeline (Python/OpenCV)
1. Capture 1920×1080 image of vertically hung garment
2. Gaussian blur (5×5) for noise reduction
3. Canny edge detection (thresholds: 50, 150)
4. Morphological dilation for edge continuity
5. Contour filtering by area (>10% image area)
6. Douglas–Peucker approximation to reduce point density
7. Pinhole camera model for pixel-to-real-world coordinate conversion
8. Discretize ironing region into vertical stroke segments

## Motion Control (Arduino)
- XY coordinates transmitted from Python via serial (9600 baud)
- Steam and vacuum activated by relay when iron reaches top of stroke
- Iron descends at 40 mm/s with steam active
- MG996R servo rotates hanger 180° after one side is complete

## Results
- 94% cloth boundary detection accuracy under standard lighting
- 40% faster than manual ironing
- Published in Scopus-indexed journal (2025)

## Hardware Components
- NEMA17 stepper motors, DRV8825 drivers, Arduino Uno, CNC Shield
- MG996R servo, relay modules, webcam (1920×1080)
- Steam iron (modified), high-speed blower for vacuum

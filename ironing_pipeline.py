"""
Ironing Robot Vision Pipeline
- GrabCut segmentation: handles white-on-white, coloured shirts, accessories (tie/hanger)
- Hanger filtered by contour aspect ratio — no blind top-crop
- Paper-aligned camera calibration
- Motion: bottom-right start → reposition UP → iron DOWN → shift LEFT → repeat
"""
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMAGE_PATH = r"C:\Desktop\OPENCV\shirt15.jpg"   # Default image; override with --image or env IRON_IMAGE (recommended so the right file always loads)

# ── Calibrated camera intrinsics ──
Z_MM = 750.0
FX   = 569.75
FY   = 568.89
CX   = 339.38
CY   = 215.55

DIST_COEFFS = np.array([-0.409, 0.133, 0.004, -0.00089, 0.037], dtype=np.float32)
CAMERA_MATRIX = np.array(
    [[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32
)

# ── End-effector width ──
END_EFFECTOR_WIDTH_MM = 30.0

# ── GrabCut margin: fraction of image shrunk inward for the GrabCut rect ──
# GrabCut needs a bounding rect that is INSIDE the background border.
# 0.03 = 3% margin from each edge → leaves a thin background border.
# Increase if your garment is very close to the image edge.
GRABCUT_MARGIN = 0.03

# ── GrabCut iterations (5 is standard; more = slower but slightly better) ──
GRABCUT_ITER = 5

# ── Morphology ──
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# ── Animation ──
IRON_STROKE_WAYPOINTS = 10
ANIM_INTERVAL_MS      = 8
SAVE_GIF              = False
GIF_FPS               = 30


# ─────────────────────────────────────────────
def mm_to_pixels(mm):
    return int(round(mm * FX / Z_MM))

STRIP_SPACING = mm_to_pixels(END_EFFECTOR_WIDTH_MM)


# ─────────────────────────────────────────────
# STEP 1 — Load & resize
# ─────────────────────────────────────────────
def load_image(path):
    path = os.path.abspath(os.path.normpath(path))
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.resize(img, (640, 480))


# ─────────────────────────────────────────────
# STEP 2 — Undistort
# ─────────────────────────────────────────────
def undistort_image(img):
    return cv2.undistort(img, CAMERA_MATRIX, DIST_COEFFS)


# ─────────────────────────────────────────────
# STEP 3 — GrabCut segmentation
#
# WHY GrabCut instead of pure HSV thresholding:
#   HSV thresholding fails on white shirts against white backgrounds
#   because there is near-zero contrast between the garment and backdrop.
#   It also picks up hangers, ties, and buttons as part of the garment
#   because they are all non-white.
#
#   GrabCut works differently — it models the COLOR DISTRIBUTION of both
#   the foreground (garment) and background (white wall) using Gaussian
#   Mixture Models, then uses a graph-cut algorithm to find the optimal
#   boundary between them. Even tiny color differences (warm white shirt
#   vs cool white backdrop) are enough for GrabCut to separate them.
#
# HOW the rect is chosen:
#   We use a fixed margin (GRABCUT_MARGIN) inset from all 4 edges.
#   This tells GrabCut: "everything in this thin border strip is
#   DEFINITELY background". Everything inside the rect is "probably
#   foreground". GrabCut then refines the exact boundary iteratively.
#
#   For a garment hung on a white backdrop this always works because:
#   - The backdrop IS the border region (always background)
#   - The garment IS somewhere in the interior (foreground)
#
# AFTER GrabCut:
#   Standard morphological cleanup removes small holes and noise.
#   Then the largest contour by area that also passes the
#   aspect-ratio filter is selected — this naturally rejects
#   the hanger rod (thin, small area) without any blind top-crop.
# ─────────────────────────────────────────────
def segment_cloth(img):
    h, w = img.shape[:2]

    # Compute the GrabCut rectangle with margin inset from all edges
    margin_x = int(w * GRABCUT_MARGIN)
    margin_y = int(h * GRABCUT_MARGIN)
    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y,
    )  # (x, y, width, height)

    # GrabCut requires a mask and two temporary arrays
    gc_mask  = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model,
                GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)

    # Pixels marked as definite foreground (3) or probable foreground (1)
    # are kept; definite background (0) and probable background (2) are removed
    cloth_rough = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                           255, 0).astype(np.uint8)

    # Morphological cleanup
    cloth_rough = cv2.morphologyEx(cloth_rough, cv2.MORPH_OPEN,  MORPH_KERNEL, iterations=2)
    cloth_rough = cv2.morphologyEx(cloth_rough, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=4)

    return cloth_rough


# ─────────────────────────────────────────────
# STEP 4 — Find garment contour
#
# Validation filters (paper specs + aspect ratio):
#   Area        > 10% of image area
#   Width       > 20% of image width
#   Height      > 20% of image height
#   Aspect ratio: width/height between 0.3 and 3.0
#     → rejects hanger rods (very thin, high aspect ratio)
#     → rejects buttons/accessories (very small area)
#
# The largest contour passing all filters is selected.
# If nothing passes, falls back to the globally largest contour
# so the pipeline never crashes.
# ─────────────────────────────────────────────
def find_garment_contour(cloth_rough, img_shape):
    edges = cv2.Canny(cloth_rough, 50, 150)
    edges = cv2.dilate(edges, MORPH_KERNEL, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(cloth_rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found. Check image and background setup.")

    h, w     = img_shape[:2]
    min_area = 0.10 * h * w
    min_w    = 0.20 * w
    min_h    = 0.20 * h

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        _, _, wc, hc = cv2.boundingRect(c)
        if wc < min_w or hc < min_h:
            continue
        aspect = wc / hc if hc > 0 else 0
        if not (0.3 <= aspect <= 3.0):
            # Reject hanger rods (very wide/thin) and accessories (very narrow)
            continue
        valid.append(c)

    if not valid:
        print("    WARNING: No contour passed filters — using largest fallback.")
        valid = contours

    largest  = max(valid, key=cv2.contourArea)
    epsilon  = 0.002 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    return simplified, edges


# ─────────────────────────────────────────────
# STEP 5 — Fill contour → garment mask
# ─────────────────────────────────────────────
def fill_contour(contour, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


# ─────────────────────────────────────────────
# STEP 6 — Pixel → real world (mm)
# X_mm = (u - cx) * Z / fx
# Y_mm = (v - cy) * Z / fy
# ─────────────────────────────────────────────
def pixel_to_real_world(px, py):
    x_mm = (px - CX) * Z_MM / FX
    y_mm = (py - CY) * Z_MM / FY
    return round(x_mm, 2), round(y_mm, 2)


# ─────────────────────────────────────────────
# STEP 7 — Generate waypoints
#
# Motion pattern (right → left):
#   Start: bottom-right of cloth
#   Even columns (0,2,4…): reposition UP   (ironing OFF)
#   Odd  columns (1,3,5…): iron stroke DOWN (ironing ON ✅)
#   After each column: shift LEFT by one EE width
# ─────────────────────────────────────────────
def generate_waypoints(mask):
    waypoints = []

    cols_with_mask = np.where(mask.any(axis=0))[0]
    if len(cols_with_mask) == 0:
        raise ValueError("Mask is empty.")

    x_right  = int(cols_with_mask[-1])
    x_left   = int(cols_with_mask[0])
    strip_xs = list(range(x_right, x_left - 1, -STRIP_SPACING))

    for col_idx, x in enumerate(strip_xs):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) == 0:
            continue

        y_top    = int(ys[0])
        y_bottom = int(ys[-1])
        rx_b, ry_b = pixel_to_real_world(x, y_bottom)
        rx_t, ry_t = pixel_to_real_world(x, y_top)

        # Starting position on very first column
        if col_idx == 0:
            waypoints.append((x, y_bottom, rx_b, ry_b, "start"))

        # Even → reposition upward (ironing OFF)
        if col_idx % 2 == 0:
            waypoints.append((x, y_bottom, rx_b, ry_b, "reposition_start"))
            waypoints.append((x, y_top,    rx_t, ry_t, "reposition_end"))

        # Odd → iron downward (ironing ON)
        else:
            iron_ys = np.linspace(y_top, y_bottom, IRON_STROKE_WAYPOINTS)
            for i, yf in enumerate(iron_ys):
                y      = int(np.clip(round(yf), y_top, y_bottom))
                rx, ry = pixel_to_real_world(x, y)
                tag    = ("iron_start" if i == 0
                          else "iron_end" if i == IRON_STROKE_WAYPOINTS - 1
                          else "iron_stroke")
                waypoints.append((x, y, rx, ry, tag))

        # Horizontal shift to next strip
        if col_idx < len(strip_xs) - 1:
            next_x  = strip_xs[col_idx + 1]
            ys_n    = np.where(mask[:, next_x] == 255)[0]
            if len(ys_n) == 0:
                continue
            shift_y = y_top if col_idx % 2 == 0 else y_bottom
            next_y  = int(ys_n[0]) if col_idx % 2 == 0 else int(ys_n[-1])
            rx0, ry0 = pixel_to_real_world(x,      shift_y)
            rx1, ry1 = pixel_to_real_world(next_x, next_y)
            waypoints.append((x,      shift_y, rx0, ry0, "horizontal_start"))
            waypoints.append((next_x, next_y,  rx1, ry1, "horizontal_end"))

    return waypoints


# ─────────────────────────────────────────────
# STEP 8 — Static pipeline visualisation
# ─────────────────────────────────────────────
def visualise_pipeline(img, cloth_rough, mask, contour, waypoints):
    img_path = img.copy()

    for i, wp in enumerate(waypoints):
        x, y, _, _, mtype = wp

        if mtype in ("iron_start", "iron_stroke") and i + 1 < len(waypoints):
            nx, ny, _, _, nt = waypoints[i + 1]
            if nt in ("iron_stroke", "iron_end"):
                cv2.line(img_path, (x, y), (nx, ny), (0, 255, 0), 2)

        elif mtype in ("reposition_start", "start") and i + 1 < len(waypoints):
            nx, ny = waypoints[i + 1][0], waypoints[i + 1][1]
            cv2.line(img_path, (x, y), (nx, ny), (80, 80, 255), 1)

        elif mtype == "horizontal_start" and i + 1 < len(waypoints):
            nx, ny = waypoints[i + 1][0], waypoints[i + 1][1]
            cv2.arrowedLine(img_path, (x, y), (nx, ny), (0, 140, 255), 1, tipLength=0.3)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Original Image\n(undistorted)")
    axes[1].imshow(cloth_rough, cmap="gray")
    axes[1].set_title("2. GrabCut Foreground Mask\n(foreground + probable foreground)")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("3. Filled Garment Mask\n(largest valid contour)")
    axes[3].imshow(cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB))
    axes[3].set_title(
        f"4. Ironing Path  [EE={END_EFFECTOR_WIDTH_MM:.0f}mm | spacing={STRIP_SPACING}px]\n"
        "green=iron↓   blue=reposition↑   orange=shift◀"
    )
    for ax in axes:
        ax.axis("off")
    plt.suptitle("Ironing Robot — Vision Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pipeline_output.png", dpi=150)
    print("Saved: pipeline_output.png")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# STEP 9 — Animation
# ─────────────────────────────────────────────
def build_animation_frames(waypoints):
    frames     = []
    iron_strip = 0
    i          = 0

    while i < len(waypoints):
        mtype = waypoints[i][4]

        if mtype == "start":
            frames.append((float(waypoints[i][0]), float(waypoints[i][1]), "idle", 0))
            i += 1

        elif mtype == "reposition_start":
            if i + 1 < len(waypoints) and waypoints[i+1][4] == "reposition_end":
                frames.append((float(waypoints[i+1][0]), float(waypoints[i+1][1]),
                               "reposition", iron_strip))
                i += 2
            else:
                i += 1

        elif mtype == "reposition_end":
            i += 1

        elif mtype in ("iron_start", "iron_stroke", "iron_end"):
            j = i
            while j < len(waypoints) and waypoints[j][4] in ("iron_start", "iron_stroke", "iron_end"):
                j += 1
            iron_strip += 1
            for k in range(i, j):
                frames.append((float(waypoints[k][0]), float(waypoints[k][1]),
                               "iron", iron_strip))
            i = j

        elif mtype == "horizontal_start":
            if i + 1 < len(waypoints) and waypoints[i+1][4] == "horizontal_end":
                frames.append((float(waypoints[i+1][0]), float(waypoints[i+1][1]),
                               "horizontal", iron_strip))
                i += 2
            else:
                i += 1

        elif mtype == "horizontal_end":
            i += 1

        else:
            i += 1

    return frames


def animate_endeffector(img, mask, waypoints):
    frames = build_animation_frames(waypoints)
    if not frames:
        print("No animation frames generated.")
        return

    total_iron = sum(1 for w in waypoints if w[4] == "iron_start")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.imshow(mask, cmap="Greens", alpha=0.12)
    ax.set_title("End-Effector Simulation\nGreen = iron stroke (↓)   Grey = reposition/shift (instant)",
                 fontsize=11)
    ax.axis("off")

    ee_dot,  = ax.plot([], [], "o", color="red", ms=9,  zorder=10)
    ee_ring, = ax.plot([], [], "o", color="red", ms=15, zorder=9,
                       mfc="none", mew=1.4, alpha=0.55)
    iron_line, = ax.plot([], [], "-", color="#00cc44", lw=2.5, zorder=5)

    iron_xs, iron_ys = [], []
    prev_seg         = None
    prev_iron_strip  = 0
    completed_iron   = []

    coord_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9,
                         color="white", va="top",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#222222", alpha=0.7))
    strip_text = ax.text(0.98, 0.96, "", transform=ax.transAxes, fontsize=9,
                         color="white", ha="right", va="top",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#222222", alpha=0.7))
    mode_text  = ax.text(0.50, 0.04, "", transform=ax.transAxes, fontsize=10,
                         color="white", ha="center",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8))

    def init():
        for a in [ee_dot, ee_ring, iron_line]:
            a.set_data([], [])
        coord_text.set_text("")
        strip_text.set_text("")
        mode_text.set_text("")
        return ee_dot, ee_ring, iron_line, coord_text, strip_text, mode_text

    def update(frame_idx):
        nonlocal iron_xs, iron_ys, prev_seg, prev_iron_strip, completed_iron

        if frame_idx >= len(frames):
            return (ee_dot, ee_ring, iron_line, coord_text, strip_text, mode_text)

        x, y, seg, strip_idx = frames[frame_idx]
        ee_dot.set_data([x], [y])
        ee_ring.set_data([x], [y])
        rx, ry = pixel_to_real_world(int(x), int(y))
        coord_text.set_text(f"X: {rx:.1f} mm   Y: {ry:.1f} mm")

        if seg == "iron":
            if strip_idx != prev_iron_strip and iron_xs:
                d, = ax.plot(iron_xs, iron_ys, "-", color="#00cc44", lw=2.5, zorder=5)
                completed_iron.append(d)
                iron_xs, iron_ys = [], []
                iron_line.set_data([], [])
            prev_iron_strip = strip_idx
            iron_xs.append(x)
            iron_ys.append(y)
            iron_line.set_data(iron_xs, iron_ys)
            ee_dot.set_color("red")
            ee_ring.set_visible(True)
            ee_ring.set_color("red")
            ee_dot.set_markersize(10)
            ee_ring.set_markersize(16)
            mode_text.set_text("▼  IRON — downward (steam + vacuum ON)")
            mode_text.set_bbox(dict(boxstyle="round,pad=0.3", fc="#004400", alpha=0.9))
            strip_text.set_text(f"Iron strip {strip_idx} / {total_iron}")

        elif seg == "reposition":
            if prev_seg == "iron" and iron_xs:
                d, = ax.plot(iron_xs, iron_ys, "-", color="#00cc44", lw=2.5, zorder=5)
                completed_iron.append(d)
                iron_xs, iron_ys = [], []
                iron_line.set_data([], [])
            ee_dot.set_color("#666666")
            ee_ring.set_visible(False)
            ee_dot.set_markersize(5)
            mode_text.set_text("▲  Reposition — instant (ironing OFF)")
            mode_text.set_bbox(dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.75))
            strip_text.set_text("")

        elif seg == "horizontal":
            if prev_seg == "iron" and iron_xs:
                d, = ax.plot(iron_xs, iron_ys, "-", color="#00cc44", lw=2.5, zorder=5)
                completed_iron.append(d)
                iron_xs, iron_ys = [], []
                iron_line.set_data([], [])
            ee_dot.set_color("#666666")
            ee_ring.set_visible(False)
            ee_dot.set_markersize(5)
            mode_text.set_text("◀  Lateral shift — instant")
            mode_text.set_bbox(dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.75))
            strip_text.set_text("")

        elif seg == "idle":
            ee_dot.set_color("yellow")
            ee_ring.set_visible(True)
            ee_ring.set_color("yellow")
            ee_dot.set_markersize(7)
            ee_ring.set_markersize(13)
            mode_text.set_text("START — bottom-right corner")
            strip_text.set_text("")

        prev_seg = seg
        return (ee_dot, ee_ring, iron_line, coord_text, strip_text, mode_text, *completed_iron)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        init_func=init,
        interval=ANIM_INTERVAL_MS,
        blit=False,
        repeat=False,
    )
    plt.tight_layout()

    if SAVE_GIF:
        print("Saving simulation.gif ...")
        ani.save("simulation.gif", writer="pillow", fps=GIF_FPS)
        print("Saved: simulation.gif")

    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ironing robot vision pipeline.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to shirt image (overrides IMAGE_PATH)")
    args = parser.parse_args()

    image_path = args.image or os.environ.get("IRON_IMAGE") or IMAGE_PATH
    image_path = os.path.abspath(os.path.normpath(image_path))

    print("=== Ironing Robot Vision Pipeline ===\n")
    print(f"[LOAD]   {image_path}")
    if not os.path.isfile(image_path):
        print(f"ERROR: File not found: {image_path}")
        print("Fix: pass --image \"C:\\\\path\\\\to\\\\shirt.jpg\" or set IRON_IMAGE env var.")
        sys.exit(1)

    print(f"[CONFIG] EE width        : {END_EFFECTOR_WIDTH_MM} mm  →  {STRIP_SPACING} px")
    print(f"[CONFIG] GrabCut margin  : {GRABCUT_MARGIN*100:.0f}% from each edge")
    print(f"[CONFIG] GrabCut iters   : {GRABCUT_ITER}")
    print(f"[CONFIG] Iron waypoints  : {IRON_STROKE_WAYPOINTS} per downward stroke\n")

    img = load_image(image_path)
    img = undistort_image(img)

    print("[3] Segmenting cloth (GrabCut)...")
    cloth_rough = segment_cloth(img)

    print("[4] Finding garment contour (Canny + aspect-ratio filter)...")
    contour, _ = find_garment_contour(cloth_rough, img.shape)
    print(f"    Contour points : {len(contour)}")

    print("[5] Filling contour mask...")
    mask = fill_contour(contour, img.shape)
    pct  = 100 * np.sum(mask == 255) / (mask.shape[0] * mask.shape[1])
    print(f"    Mask coverage  : {pct:.1f}%")

    print("[6] Generating waypoints...")
    waypoints   = generate_waypoints(mask)
    iron_count  = sum(1 for w in waypoints if w[4] == "iron_start")
    repos_count = sum(1 for w in waypoints if w[4] == "reposition_start")
    print(f"    Iron strips    : {iron_count}  (downward, steam+vacuum ON)")
    print(f"    Reposition     : {repos_count}  (upward, ironing OFF)")

    print("\n    Sample waypoints:")
    for wp in waypoints[:10]:
        print(f"      px=({wp[0]:4d},{wp[1]:4d})  X={wp[2]:8.2f}mm  Y={wp[3]:8.2f}mm  [{wp[4]}]")

    print("\n[7] Pipeline visualisation...")
    visualise_pipeline(img, cloth_rough, mask, contour, waypoints)

    print("[8] Animation...")
    animate_endeffector(img, mask, waypoints)

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()

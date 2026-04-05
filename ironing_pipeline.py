"""
Ironing Robot — Vision + Serial Pipeline
Cartesian-Based Automatic Ironing Robot | ICAIT 2025
"""

import argparse, os, sys, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CAMERA_INDEX   = 0
CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480

Z_MM = 750.0
FX, FY, CX, CY = 569.75, 568.89, 339.38, 215.55

DIST_COEFFS   = np.array([-0.409, 0.133, 0.004, -0.00089, 0.037], dtype=np.float32)
CAMERA_MATRIX = np.array([[FX,0,CX],[0,FY,CY],[0,0,1]], dtype=np.float32)

END_EFFECTOR_WIDTH_MM = 30.0
GRABCUT_MARGIN        = 0.03
GRABCUT_ITER          = 5
MORPH_KERNEL          = np.ones((5,5), np.uint8)

SERIAL_ENABLED        = False
SERIAL_PORT           = "COM3"
SERIAL_BAUD           = 115200
SERIAL_TIMEOUT        = 10

STEPS_PER_MM          = 20
FEED_RATE_MM_MIN      = 3000
IRON_FEED_RATE        = 1500
IRON_STROKE_WAYPOINTS = 10

ANIM_INTERVAL_MS      = 8
SAVE_GIF              = False
GIF_FPS               = 30

STRIP_SPACING = int(round(END_EFFECTOR_WIDTH_MM * FX / Z_MM))


# ═════════════════════════════════════════════
# SECTION 1 — IMAGE ACQUISITION
# ═════════════════════════════════════════════

def capture_from_webcam():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    print("\n[CAMERA] Press SPACE to capture, ESC to cancel.\n")
    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, "SPACE=capture  ESC=cancel",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Camera Preview", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            captured = cv2.resize(frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
            print("[CAMERA] Captured.")
            break
        elif key == 27:
            print("[CAMERA] Cancelled.")
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured


def load_image_from_file(path):
    path = os.path.abspath(os.path.normpath(path))
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.resize(img, (CAPTURE_WIDTH, CAPTURE_HEIGHT))


# ═════════════════════════════════════════════
# SECTION 2 — IMAGE PROCESSING
# ═════════════════════════════════════════════

def undistort_image(img):
    return cv2.undistort(img, CAMERA_MATRIX, DIST_COEFFS)


def segment_cloth(img):
    """GrabCut segmentation — handles white shirts on white backdrops."""
    h, w = img.shape[:2]
    mx, my = int(w * GRABCUT_MARGIN), int(h * GRABCUT_MARGIN)
    rect  = (mx, my, w - 2*mx, h - 2*my)

    gc_mask   = np.zeros((h,w), dtype=np.uint8)
    bgd_model = np.zeros((1,65), dtype=np.float64)
    fgd_model = np.zeros((1,65), dtype=np.float64)

    cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model,
                GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)

    cloth_rough = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    cloth_rough = cv2.morphologyEx(cloth_rough, cv2.MORPH_OPEN,  MORPH_KERNEL, iterations=2)
    cloth_rough = cv2.morphologyEx(cloth_rough, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=4)
    return cloth_rough


def find_garment_contour(cloth_rough, img_shape):
    """Canny + Douglas-Peucker. Filters by area, size, aspect ratio (rejects hanger)."""
    edges = cv2.Canny(cloth_rough, 50, 150)
    edges = cv2.dilate(edges, MORPH_KERNEL, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(cloth_rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found.")

    h, w = img_shape[:2]
    valid = []
    for c in contours:
        if cv2.contourArea(c) < 0.10 * h * w:
            continue
        _, _, wc, hc = cv2.boundingRect(c)
        if wc < 0.20*w or hc < 0.20*h:
            continue
        if not (0.3 <= wc/hc <= 3.0):
            continue
        valid.append(c)

    if not valid:
        print("    WARNING: No contour passed filters — using largest fallback.")
        valid = contours

    largest  = max(valid, key=cv2.contourArea)
    epsilon  = 0.002 * cv2.arcLength(largest, True)
    return cv2.approxPolyDP(largest, epsilon, True), edges


def fill_contour(contour, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


# ═════════════════════════════════════════════
# SECTION 3 — COORDINATE CONVERSION
# ═════════════════════════════════════════════

def pixel_to_real_world(px, py):
    """Pinhole model: X_mm = (u-cx)*Z/fx,  Y_mm = (v-cy)*Z/fy"""
    return round((px-CX)*Z_MM/FX, 2), round((py-CY)*Z_MM/FY, 2)


def real_world_to_steps(x_mm, y_mm):
    return int(round(x_mm * STEPS_PER_MM)), int(round(y_mm * STEPS_PER_MM))


# ═════════════════════════════════════════════
# SECTION 4 — WAYPOINT GENERATION
# ═════════════════════════════════════════════

def generate_waypoints(mask):
    """
    Boustrophedon trajectory, right → left:
      Even col: reposition UP  (ironing OFF)
      Odd  col: iron DOWN      (steam + vacuum ON)
    Tags: start | reposition_start/end | iron_start/stroke/end | horizontal_start/end
    """
    waypoints = []
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        raise ValueError("Mask is empty.")

    strip_xs = list(range(int(cols[-1]), int(cols[0])-1, -STRIP_SPACING))

    for col_idx, x in enumerate(strip_xs):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) == 0:
            continue
        y_top, y_bottom = int(ys[0]), int(ys[-1])
        rx_b, ry_b = pixel_to_real_world(x, y_bottom)
        rx_t, ry_t = pixel_to_real_world(x, y_top)

        if col_idx == 0:
            waypoints.append((x, y_bottom, rx_b, ry_b, "start"))

        if col_idx % 2 == 0:
            waypoints.append((x, y_bottom, rx_b, ry_b, "reposition_start"))
            waypoints.append((x, y_top,    rx_t, ry_t, "reposition_end"))
        else:
            for i, yf in enumerate(np.linspace(y_top, y_bottom, IRON_STROKE_WAYPOINTS)):
                y = int(np.clip(round(yf), y_top, y_bottom))
                rx, ry = pixel_to_real_world(x, y)
                tag = ("iron_start" if i == 0 else
                       "iron_end"   if i == IRON_STROKE_WAYPOINTS-1 else
                       "iron_stroke")
                waypoints.append((x, y, rx, ry, tag))

        if col_idx < len(strip_xs)-1:
            nx  = strip_xs[col_idx+1]
            ysn = np.where(mask[:, nx] == 255)[0]
            if len(ysn) == 0:
                continue
            sy  = y_top    if col_idx % 2 == 0 else y_bottom
            ny  = int(ysn[0]) if col_idx % 2 == 0 else int(ysn[-1])
            waypoints.append((x,  sy, *pixel_to_real_world(x,  sy), "horizontal_start"))
            waypoints.append((nx, ny, *pixel_to_real_world(nx, ny), "horizontal_end"))

    return waypoints


# ═════════════════════════════════════════════
# SECTION 5 — SERIAL COMMUNICATION
# ═════════════════════════════════════════════

class RobotSerial:
    """
    Handshake protocol: Python sends command → Arduino executes → replies "OK".
    Commands: MOVE X Y F | STEAM_ON/OFF | VACUUM_ON/OFF | SERVO_ROTATE | HOME
    """
    def __init__(self, port, baud=115200, timeout=10):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("Run: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        self.ser.flushInput()
        print(f"[SERIAL] Connected on {port} at {baud} baud.")

    def send(self, command):
        self.ser.write((command.strip() + "\n").encode("utf-8"))
        deadline = time.time() + SERIAL_TIMEOUT
        while time.time() < deadline:
            if self.ser.in_waiting:
                response = self.ser.readline().decode("utf-8").strip()
                if response == "OK":
                    return True
                elif response.startswith("ERR"):
                    print(f"  [SERIAL] {response}")
                    return False
        print(f"  [SERIAL] Timeout: {command}")
        return False

    def move(self, x, y, feed=None):
        return self.send(f"MOVE X{x:.2f} Y{y:.2f} F{feed or FEED_RATE_MM_MIN}")
    def steam_on(self):    return self.send("STEAM_ON")
    def steam_off(self):   return self.send("STEAM_OFF")
    def vacuum_on(self):   return self.send("VACUUM_ON")
    def vacuum_off(self):  return self.send("VACUUM_OFF")
    def servo_rotate(self):return self.send("SERVO_ROTATE")
    def home(self):        return self.send("HOME")
    def close(self):
        self.ser.close()
        print("[SERIAL] Closed.")


def execute_waypoints(waypoints, robot):
    print("\n[ROBOT] Starting ironing sequence...")
    total_iron = sum(1 for w in waypoints if w[4] == "iron_start")
    iron_count = 0

    for wp in waypoints:
        _, _, rx, ry, tag = wp

        if tag == "start":
            robot.move(rx, ry)
        elif tag == "reposition_start":
            robot.move(rx, ry, feed=FEED_RATE_MM_MIN)
        elif tag == "reposition_end":
            pass
        elif tag == "iron_start":
            iron_count += 1
            print(f"  [IRON {iron_count}/{total_iron}] STEAM + VACUUM ON")
            robot.steam_on(); robot.vacuum_on()
            robot.move(rx, ry, feed=IRON_FEED_RATE)
        elif tag in ("iron_stroke",):
            robot.move(rx, ry, feed=IRON_FEED_RATE)
        elif tag == "iron_end":
            robot.move(rx, ry, feed=IRON_FEED_RATE)
            robot.steam_off(); robot.vacuum_off()
            print(f"  [IRON {iron_count}/{total_iron}] STEAM + VACUUM OFF")
        elif tag == "horizontal_start":
            robot.move(rx, ry, feed=FEED_RATE_MM_MIN)
        elif tag == "horizontal_end":
            pass

    print("\n[ROBOT] Side 1 complete. Rotating hanger...")
    robot.servo_rotate()
    robot.home()
    print("[ROBOT] Done.")


# ═════════════════════════════════════════════
# SECTION 6 — VISUALISATION
# ═════════════════════════════════════════════

def visualise_pipeline(img, cloth_rough, mask, contour, waypoints):
    img_path = img.copy()
    for i, wp in enumerate(waypoints):
        x, y, _, _, t = wp
        if t in ("iron_start","iron_stroke") and i+1 < len(waypoints):
            nx, ny, _, _, nt = waypoints[i+1]
            if nt in ("iron_stroke","iron_end"):
                cv2.line(img_path, (x,y), (nx,ny), (0,255,0), 2)
        elif t in ("reposition_start","start") and i+1 < len(waypoints):
            cv2.line(img_path, (x,y), (waypoints[i+1][0],waypoints[i+1][1]), (80,80,255), 1)
        elif t == "horizontal_start" and i+1 < len(waypoints):
            cv2.arrowedLine(img_path, (x,y),
                            (waypoints[i+1][0],waypoints[i+1][1]), (0,140,255), 1, tipLength=0.3)

    fig, axes = plt.subplots(1, 4, figsize=(22,6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));  axes[0].set_title("1. Captured (undistorted)")
    axes[1].imshow(cloth_rough, cmap="gray");              axes[1].set_title("2. GrabCut Mask")
    axes[2].imshow(mask, cmap="gray");                     axes[2].set_title("3. Filled Garment Mask")
    axes[3].imshow(cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"4. Ironing Path [EE={END_EFFECTOR_WIDTH_MM:.0f}mm]\ngreen=iron↓  blue=reposition↑  orange=shift◀")
    for ax in axes: ax.axis("off")
    plt.suptitle("Ironing Robot — Vision Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pipeline_output.png", dpi=150)
    print("Saved: pipeline_output.png")
    plt.show(); plt.close(fig)


# ═════════════════════════════════════════════
# SECTION 7 — ANIMATION
# ═════════════════════════════════════════════

def build_animation_frames(waypoints):
    frames, iron_strip, i = [], 0, 0
    while i < len(waypoints):
        t = waypoints[i][4]
        if t == "start":
            frames.append((float(waypoints[i][0]), float(waypoints[i][1]), "idle", 0)); i+=1
        elif t == "reposition_start" and i+1 < len(waypoints) and waypoints[i+1][4] == "reposition_end":
            frames.append((float(waypoints[i+1][0]), float(waypoints[i+1][1]), "reposition", iron_strip)); i+=2
        elif t == "reposition_end":
            i+=1
        elif t in ("iron_start","iron_stroke","iron_end"):
            j = i
            while j < len(waypoints) and waypoints[j][4] in ("iron_start","iron_stroke","iron_end"): j+=1
            iron_strip += 1
            for k in range(i,j):
                frames.append((float(waypoints[k][0]), float(waypoints[k][1]), "iron", iron_strip))
            i = j
        elif t == "horizontal_start" and i+1 < len(waypoints) and waypoints[i+1][4] == "horizontal_end":
            frames.append((float(waypoints[i+1][0]), float(waypoints[i+1][1]), "horizontal", iron_strip)); i+=2
        elif t == "horizontal_end":
            i+=1
        else:
            i+=1
    return frames


def animate_endeffector(img, mask, waypoints):
    frames = build_animation_frames(waypoints)
    if not frames:
        print("No frames generated."); return

    total_iron = sum(1 for w in waypoints if w[4] == "iron_start")
    fig, ax = plt.subplots(figsize=(9,7))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.imshow(mask, cmap="Greens", alpha=0.12)
    ax.set_title("End-Effector Simulation\nGreen=iron↓   Grey=reposition/shift", fontsize=11)
    ax.axis("off")

    ee_dot,  = ax.plot([], [], "o", color="red", ms=9,  zorder=10)
    ee_ring, = ax.plot([], [], "o", color="red", ms=15, zorder=9, mfc="none", mew=1.4, alpha=0.55)
    iron_line,= ax.plot([], [], "-", color="#00cc44", lw=2.5, zorder=5)

    iron_xs, iron_ys, prev_seg, prev_strip, done = [], [], None, 0, []

    coord_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, color="white", va="top",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#222222", alpha=0.7))
    strip_text = ax.text(0.98, 0.96, "", transform=ax.transAxes, fontsize=9, color="white",
                         ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", fc="#222222", alpha=0.7))
    mode_text  = ax.text(0.50, 0.04, "", transform=ax.transAxes, fontsize=10, color="white",
                         ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8))

    def init():
        for a in [ee_dot, ee_ring, iron_line]: a.set_data([], [])
        coord_text.set_text(""); strip_text.set_text(""); mode_text.set_text("")
        return ee_dot, ee_ring, iron_line, coord_text, strip_text, mode_text

    def update(fi):
        nonlocal iron_xs, iron_ys, prev_seg, prev_strip, done
        if fi >= len(frames):
            return (ee_dot, ee_ring, iron_line, coord_text, strip_text, mode_text)
        x, y, seg, sidx = frames[fi]
        ee_dot.set_data([x],[y]); ee_ring.set_data([x],[y])
        rx, ry = pixel_to_real_world(int(x), int(y))
        coord_text.set_text(f"X: {rx:.1f} mm   Y: {ry:.1f} mm")

        if seg == "iron":
            if sidx != prev_strip and iron_xs:
                d, = ax.plot(iron_xs, iron_ys, "-", color="#00cc44", lw=2.5, zorder=5)
                done.append(d); iron_xs, iron_ys = [], []; iron_line.set_data([], [])
            prev_strip = sidx
            iron_xs.append(x); iron_ys.append(y); iron_line.set_data(iron_xs, iron_ys)
            ee_dot.set_color("red"); ee_ring.set_visible(True); ee_ring.set_color("red")
            ee_dot.set_markersize(10); ee_ring.set_markersize(16)
            mode_text.set_text("▼  IRON — downward (steam + vacuum ON)")
            mode_text.set_bbox(dict(boxstyle="round,pad=0.3", fc="#004400", alpha=0.9))
            strip_text.set_text(f"Iron strip {sidx} / {total_iron}")
        elif seg in ("reposition", "horizontal"):
            if prev_seg == "iron" and iron_xs:
                d, = ax.plot(iron_xs, iron_ys, "-", color="#00cc44", lw=2.5, zorder=5)
                done.append(d); iron_xs, iron_ys = [], []; iron_line.set_data([], [])
            ee_dot.set_color("#666666"); ee_ring.set_visible(False); ee_dot.set_markersize(5)
            mode_text.set_text("▲  Reposition" if seg=="reposition" else "◀  Lateral shift")
            mode_text.set_bbox(dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.75))
            strip_text.set_text("")
        elif seg == "idle":
            ee_dot.set_color("yellow"); ee_ring.set_visible(True); ee_ring.set_color("yellow")
            ee_dot.set_markersize(7); ee_ring.set_markersize(13)
            mode_text.set_text("START — bottom-right corner"); strip_text.set_text("")

        prev_seg = seg
        return (ee_dot, ee_ring, iron_line, coord_text, strip_text, mode_text, *done)

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                   init_func=init, interval=ANIM_INTERVAL_MS,
                                   blit=False, repeat=False)
    plt.tight_layout()
    if SAVE_GIF:
        print("Saving simulation.gif ...")
        ani.save("simulation.gif", writer="pillow", fps=GIF_FPS)
        print("Saved: simulation.gif")
    plt.show()


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ironing Robot Vision + Serial Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Image path (skips webcam)")
    parser.add_argument("--run",   action="store_true",    help="Execute on hardware via serial")
    args = parser.parse_args()

    print("=== Ironing Robot Vision Pipeline ===\n")
    print(f"[CONFIG] EE width={END_EFFECTOR_WIDTH_MM}mm  strip={STRIP_SPACING}px  "
          f"steps/mm={STEPS_PER_MM}  serial={'ON' if SERIAL_ENABLED else 'OFF'}\n")

    img = load_image_from_file(args.image) if args.image else capture_from_webcam()
    if img is None:
        print("No image. Exiting."); sys.exit(0)

    print("[2] Undistorting..."); img = undistort_image(img)
    print("[3] GrabCut segmentation..."); cloth_rough = segment_cloth(img)
    print("[4] Finding contour..."); contour, _ = find_garment_contour(cloth_rough, img.shape)
    print(f"    Contour points: {len(contour)}")
    print("[5] Filling mask..."); mask = fill_contour(contour, img.shape)
    print(f"    Coverage: {100*np.sum(mask==255)/(mask.shape[0]*mask.shape[1]):.1f}%")
    print("[6] Generating waypoints..."); waypoints = generate_waypoints(mask)

    iron_n  = sum(1 for w in waypoints if w[4]=="iron_start")
    repos_n = sum(1 for w in waypoints if w[4]=="reposition_start")
    print(f"    Iron strips: {iron_n}  |  Repositions: {repos_n}  |  Total WP: {len(waypoints)}")

    print("\n    Sample (pixel → mm → steps):")
    for wp in waypoints[:8]:
        sx, sy = real_world_to_steps(wp[2], wp[3])
        print(f"      px=({wp[0]:4d},{wp[1]:4d})  {wp[2]:8.2f}mm {wp[3]:8.2f}mm  "
              f"steps=({sx:6d},{sy:6d})  [{wp[4]}]")

    print("\n[7] Visualising pipeline...")
    visualise_pipeline(img, cloth_rough, mask, contour, waypoints)

    if args.run and SERIAL_ENABLED:
        print("\n[8] Executing on hardware...")
        robot = RobotSerial(SERIAL_PORT, SERIAL_BAUD)
        try:
            execute_waypoints(waypoints, robot)
        finally:
            robot.close()
    else:
        if args.run:
            print("\n[8] SERIAL_ENABLED=False — running simulation instead.")
        print("\n[8] Animating end-effector...")
        animate_endeffector(img, mask, waypoints)

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()

  

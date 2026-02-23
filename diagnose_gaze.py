"""
CogniGaze diagnostic — measures raw iris coords at 5 gaze positions.

Run:  python diagnose_gaze.py
Then follow the on-screen prompts (look at each position, press SPACE).
"""

import cv2
import mediapipe as mp
import sys

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

RIGHT_IRIS_CENTER = 473

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera 0"); sys.exit(1)

POSITIONS = ["centre", "left", "right", "top", "bottom"]
INSTRUCTIONS = {
    "centre": "Look STRAIGHT at screen centre",
    "left":   "Look as FAR LEFT as comfortable",
    "right":  "Look as FAR RIGHT as comfortable",
    "top":    "Look as FAR UP as comfortable",
    "bottom": "Look as FAR DOWN as comfortable",
}

results = {}

def get_iris_raw(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    if len(lm) <= RIGHT_IRIS_CENTER:
        return None
    iris = lm[RIGHT_IRIS_CENTER]
    return (iris.x, iris.y)

print("\n=== CogniGaze Gaze Diagnostic ===")
print("Press SPACE to record each position, Q to quit.\n")

for pos in POSITIONS:
    print(f"[{pos.upper()}] {INSTRUCTIONS[pos]}")
    print("  Hold still, then press SPACE ...")

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        h, w = frame.shape[:2]
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            if len(lm) > RIGHT_IRIS_CENTER:
                ix = int(lm[RIGHT_IRIS_CENTER].x * w)
                iy = int(lm[RIGHT_IRIS_CENTER].y * h)
                cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)
                cv2.putText(frame,
                    f"iris_x={lm[RIGHT_IRIS_CENTER].x:.4f}  iris_y={lm[RIGHT_IRIS_CENTER].y:.4f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"{pos.upper()}: {INSTRUCTIONS[pos]}",
            (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        cv2.putText(frame, "SPACE = record  |  Q = quit",
            (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("CogniGaze Diagnostic", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cam.release(); cv2.destroyAllWindows(); sys.exit(0)
        if key == ord(' '):
            print(f"  Recording 30 frames for [{pos}]...")
            samples = []
            for _ in range(30):
                ok2, f2 = cam.read()
                if not ok2: continue
                pt = get_iris_raw(f2)
                if pt:
                    samples.append(pt)
            if samples:
                mx = sum(s[0] for s in samples) / len(samples)
                my = sum(s[1] for s in samples) / len(samples)
                results[pos] = (mx, my)
                print(f"  Recorded: iris_x={mx:.4f}  iris_y={my:.4f}  (n={len(samples)})")
                break
            else:
                print("  No iris detected — try again")

cam.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("  RAW IRIS RESULTS")
print("="*60)
cx, cy = results["centre"]
print(f"\nCentre:  iris_x={cx:.4f}  iris_y={cy:.4f}\n")

for pos in ["left","right","top","bottom"]:
    px, py = results[pos]
    dx = px - cx
    dy = py - cy
    print(f"{pos:8s}: iris_x={px:.4f}  iris_y={py:.4f}   dx={dx:+.4f}  dy={dy:+.4f}")

lx, _ = results["left"];  rx, _ = results["right"]
_, ty = results["top"];   _, by = results["bottom"]
x_travel = rx - lx
y_travel = by - ty

print(f"\nHorizontal travel (right-left): {x_travel:+.4f}")
print(f"Vertical   travel (down-top):   {y_travel:+.4f}")

print("\n── Direction check ──")
cx2, cy2 = results["centre"]
rx2, _ = results["right"]; lx2, _ = results["left"]
_, by2 = results["bottom"]; _, ty2 = results["top"]

x_ok = rx2 > cx2 and lx2 < cx2
y_ok = by2 > cy2 and ty2 < cy2
print(f"  X (right increases): {'✓ correct' if x_ok else '✗ INVERTED — negate dx in main.py'}")
print(f"  Y (down  increases): {'✓ correct' if y_ok else '✗ INVERTED — negate dy in main.py'}")

print("\n── Suggested config values ──")
if abs(x_travel) > 0.005:
    gx = round(abs(0.85 / x_travel), 1)
    print(f"  relative_iris_gain_x = {gx}")
else:
    print("  WARNING: X travel too small")

if abs(y_travel) > 0.003:
    gy = round(abs(0.85 / y_travel), 1)
    print(f"  relative_iris_gain_y = {gy}")
else:
    print("  WARNING: Y travel too small")

print(f"\n  dx formula: {'-(iris_x - cx)' if not x_ok else '(iris_x - cx)'}")
print(f"  dy formula: {'-(iris_y - cy)' if not y_ok else '(iris_y - cy)'}")
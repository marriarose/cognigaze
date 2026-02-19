# Cognigaze – Hybrid Predictive Gaze Interaction System

This project uses OpenCV, Mediapipe, and PyAutoGUI to control the mouse cursor with eye movements. It leverages facial landmark detection to track eye movements and translate them into mouse actions.

## Launch paths

- **Production (modular system):** From the repository root, run `python launch_cognigaze.py` to start the hybrid predictive gaze system (`eye_gaze_system` package). This is the main entry point.
- **Legacy prototypes:** The scripts in this folder (`cognigaze_eye_interface.py`, `cognigaze_gesture_interface.py`) are transitional modules and run only when explicitly invoked (e.g. `python cognigaze_eye_interface.py` from this directory).

## Prerequisites

- Python 3.x
- OpenCV
- Mediapipe
- PyAutoGUI

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/cognigaze/cognigaze.git
    cd cognigaze
    ```

2. Install the required libraries:

    ```sh
    pip install opencv-python mediapipe pyautogui
    ```

## Usage

1. Run the script:

    ```sh
    python cognigaze_eye_interface.py
    ```

2. The script will initialize your webcam and start detecting your facial landmarks. The mouse cursor will move based on the specified facial landmarks.

3. To right-click, close your left eye. The script detects this by checking if the difference in y-coordinates of specific left eye landmarks is less than a threshold.

4. Press `q` to exit the program.

## Code Explanation

### Import necessary libraries

```python
import cv2
import mediapipe as mp
import pyautogui
```

### Initialize the camera and Face Mesh from Mediapipe

```python
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
```

### Get the screen size for mouse control

```python
screen_w, screen_h = pyautogui.size()
```

### Main loop for capturing and processing video frames

```python
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 300, 300))

        if (left[0].y - left[1].y) < 0.004:
            pyautogui.rightClick()
            pyautogui.sleep(1)

    cv2.imshow('Cognigaze – Hybrid Predictive Gaze Interaction System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

### Explanation

- The script captures video frames from the webcam and flips them horizontally for a better user experience.
- The frames are converted to RGB format and processed using Mediapipe's Face Mesh to detect facial landmarks.
- Specific facial landmarks are tracked to control the mouse cursor.
- If the difference in y-coordinates of specific left eye landmarks is less than a threshold, a right-click action is triggered.
- The script displays the frame with annotations and waits for a key press event to exit the loop.

## Limitations

Eye gaze tracking systems have inherent constraints. Understanding these helps set realistic expectations and troubleshoot issues.

### Lighting Sensitivity

- **Effect**: Face and iris detection depend on consistent, even lighting. Low light reduces landmark visibility; strong shadows or backlight can cause incorrect or lost tracking.
- **Mitigation**: Use stable, diffuse lighting. Avoid bright windows or lamps directly behind you. A ring light or soft front light can improve reliability.

### Vertical Resolution Limitations

- **Effect**: Vertical gaze (up/down) is generally less accurate than horizontal. MediaPipe and similar models are optimized for frontal faces; looking up or down changes the face angle and degrades landmark quality.
- **Mitigation**: Keep the face roughly frontal. Calibration can partly compensate, but vertical precision will remain lower than horizontal.

### Parallax Issues

- **Effect**: The camera and the user’s eyes are not at the same position. Gaze direction in camera space does not match true gaze direction, which can cause systematic offsets, especially at screen edges.
- **Mitigation**: Use calibration (5-point grid) to learn a mapping between detected gaze and screen coordinates. Results vary with head pose and viewing distance.

### Calibration Drift

- **Effect**: Over time, the mapping between gaze and cursor can shift due to head movement, posture changes, fatigue, or lighting changes. Calibration is not permanently stable.
- **Mitigation**: Recalibrate periodically or when accuracy noticeably degrades. Maintain consistent head position and viewing distance during use.

### Data Loss Scenarios

Tracking can be lost or degraded in these situations:

- **Face out of frame**: Camera no longer sees the face.
- **Partial occlusion**: Hands, hair, glasses, or objects covering the eyes.
- **Rapid head motion**: Motion blur or landmark instability.
- **Frame drops**: Camera or processing lag causes missed frames.
- **Low confidence**: Detection uncertainty below the confidence threshold.

The system responds by **freezing the cursor** at the last valid position instead of jumping, and attempts **fallback pupil detection** when MediaPipe confidence drops.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

# Cognigaze Gesture Interface

This project uses OpenCV, Mediapipe, and PyAutoGUI to control the mouse cursor and perform actions with hand gestures. It leverages hand landmark detection to translate gestures into mouse movements and clicks.

## Prerequisites

- Python 3.x
- OpenCV
- Mediapipe
- PyAutoGUI

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/cognigaze/cognigaze.git
    cd cognigaze
    ```

2. Install the required libraries:

    ```sh
    pip install opencv-python mediapipe pyautogui
    ```

## Usage

1. Run the script:

    ```sh
    python cognigaze_gesture_interface.py
    ```

2. The script will initialize your webcam and start detecting hand landmarks. The mouse cursor will move based on the specified hand landmarks.

3. Hand Gestures:
    - **Move Cursor**: Use your index finger to move the cursor.
    - **Left Click**: Bring your thumb close to your index finger.
    - **Scroll**: Increase the distance between your thumb and index finger to scroll up; decrease the distance to scroll down.
    - **Right Click**: Lower your middle finger below your index finger.

4. Press `q` to exit the program.

## Code Explanation

### Import necessary libraries

```python
import cv2
import mediapipe as mp
import pyautogui
import math
```

### Disable PyAutoGUI fail-safe

```python
pyautogui.FAILSAFE = False
```

### Initialize the camera and hand detector

```python
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_x, index_y = 0, 0
middle_finger_clicked = False
```

### Main loop for capturing and processing video frames

```python
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=20, color=(100, 255, 255))
                    index_x = int(screen_width / frame_width * x)
                    index_y = int(screen_height / frame_height * y)

                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=20, color=(100, 255, 255))
                    thumb_x = int(screen_width / frame_width * x)
                    thumb_y = int(screen_height / frame_height * y)

                    # Move the mouse cursor
                    pyautogui.moveTo(index_x, index_y)

                    # Perform a click if fingers are close
                    if abs(index_y - thumb_y) < 20:
                        pyautogui.mouseDown()
                        pyautogui.sleep(1)  # Adjust the duration as needed
                        pyautogui.mouseUp()

                    # Scroll up or down based on the distance between thumb and index finger
                    distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                    if distance > 100:
                        pyautogui.scroll(1)  # Scroll up
                    elif distance < 50:
                        pyautogui.scroll(-1)  # Scroll down

                if id == 12:  # Assuming middle finger
                    if landmarks[8].y < landmarks[12].y:  # Check if middle finger is below index finger
                        middle_finger_clicked = True
                    else:
                        middle_finger_clicked = False

                    # Perform right-click if middle finger is below index finger
                    if middle_finger_clicked:
                        pyautogui.rightClick()

    cv2.imshow('Cognigaze Gesture Interface', frame)
    cv2.waitKey(1)
```

### Explanation

- The script captures video frames from the webcam and flips them horizontally for a better user experience.
- The frames are converted to RGB format and processed using Mediapipe's Hand detector to detect hand landmarks.
- Specific hand landmarks are tracked to control the mouse cursor and perform actions.
- The script displays the frame with annotations and waits for a key press event to exit the loop.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---


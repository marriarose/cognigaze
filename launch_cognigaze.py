"""Entry script to launch the Cognigaze eye gaze system.

This is the single entry point for the production modular hybrid gaze system.
Legacy scripts (cognigaze_eye_interface, cognigaze_gesture_interface) do not
auto-run; they execute only when explicitly invoked.
"""

from eye_gaze_system.main import main

if __name__ == '__main__':
    main()

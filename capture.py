import cv2
import numpy as np
import mss

def capture_screen(region=None):
    """
    Capture a portion of the screen or the entire screen.

    :param region: Tuple (left, top, width, height) defining the region to capture.
                   If None, captures the entire screen.
    :return: Captured screen image as a NumPy array (BGR format).
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor

        if region:
            monitor = {"top": region[1], "left": region[0], "width": region[2], "height": region[3]}

        screenshot = np.array(sct.grab(monitor))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

if __name__ == "__main__":
    # Test the screen capture
    while True:
        frame = capture_screen(region=(0, 0, 800, 600))  # Example region
        cv2.imshow("Screen Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()

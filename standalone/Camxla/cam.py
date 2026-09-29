"""
Standalone test for RobotVisionV2.

Pipeline:

    Camera
       |
       v
 CameraStream
       |
       v
 BGR Frame
       |
       v
 RobotVisionV2.predict_gpu()
       |
       v
 DetectedObject
       |
       v
 Draw Bounding Box
       |
       v
 cv2.imshow()

No UART.
No YAML.
No robot control.
"""

import cv2
import time
import logging

from camera import CameraStream
from vision_v2 import RobotVisionV2


# ============================================================
# CONFIG
# ============================================================

# Camera device
CAMERA_ID = 0

# TensorRT engine.
#
# Có thể trỏ trực tiếp tới:
#
#     "/home/user/model/best.engine"
#
# hoặc thư mục:
#
#     "/home/user/model"
#
# Nếu truyền thư mục, RobotVisionV2 sẽ tự tìm:
#
#     <MODEL_PATH>/best.engine
#
MODEL_PATH = "/path/to/your/model"

# Input size của model
IMAGE_SIZE = 640

# Detection confidence threshold
CONF_THRESHOLD = 0.5

# Camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(frame, detections):
    """
    Draw detections lên frame.

    LƯU Ý:
    Hàm này cần biết chính xác cấu trúc của DetectedObject
    trong vision.py.

    Hiện tại RobotVisionV2 trả về:
        List[DetectedObject]

    Nếu DetectedObject của project có các field:
        x1, y1, x2, y2
        confidence
        class_id

    thì đoạn dưới sẽ hoạt động.

    Nếu tên field khác, cần sửa theo vision.py.
    """

    for detection in detections:

        try:
            # ------------------------------------------------
            # Lấy bounding box
            # ------------------------------------------------

            x1 = int(detection.x1)
            y1 = int(detection.y1)
            x2 = int(detection.x2)
            y2 = int(detection.y2)

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(detection.confidence)

            # ------------------------------------------------
            # Class ID
            # ------------------------------------------------

            class_id = int(detection.class_id)

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = "class={} {:.2f}".format(
                class_id,
                confidence,
            )

            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )

            text_y = max(y1, text_height + 5)

            cv2.rectangle(
                frame,
                (x1, text_y - text_height - baseline - 5),
                (x1 + text_width + 5, text_y),
                (0, 255, 0),
                -1,
            )

            # ------------------------------------------------
            # Label text
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (x1 + 2, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        except AttributeError:

            # ------------------------------------------------
            # Nếu cấu trúc DetectedObject khác dự kiến,
            # không làm crash cả chương trình.
            # ------------------------------------------------

            logging.warning(
                "DetectedObject khong co field "
                "x1/y1/x2/y2/confidence/class_id."
            )

            logging.warning(
                "Detection = %r",
                detection,
            )

            break

        except Exception as e:

            logging.warning(
                "Cannot draw detection: %s",
                e,
            )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RobotVision V2 - Standalone Vision Test")
    print("=" * 60)

    camera = None
    detector = None

    try:

        # ====================================================
        # 1. OPEN CAMERA
        # ====================================================

        print("[INFO] Opening camera...")

        camera = CameraStream(
            src=CAMERA_ID,
            buffer_size=1,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
        )

        camera.start()

        print("[INFO] Camera started.")

        # ====================================================
        # 2. LOAD ROBOT VISION V2
        # ====================================================

        print("[INFO] Loading RobotVisionV2...")

        detector = RobotVisionV2(
            model_path=MODEL_PATH,
            imgsz=IMAGE_SIZE,
            device="GPU",
        )

        print("[INFO] RobotVisionV2 loaded.")

        # ====================================================
        # 3. CREATE WINDOW
        # ====================================================

        window_name = "RobotVision V2"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL,
        )

        # Có thể resize cửa sổ
        cv2.resizeWindow(
            window_name,
            1280,
            720,
        )

        # ====================================================
        # 4. FPS
        # ====================================================

        prev_time = time.time()

        frame_count = 0

        # ====================================================
        # 5. MAIN LOOP
        # ====================================================

        while True:

            # ------------------------------------------------
            # GET LATEST FRAME
            # ------------------------------------------------

            frame = camera.read()

            if frame is None:
                continue

            frame_count += 1

            # ------------------------------------------------
            # DETECTION
            # ------------------------------------------------

            inference_start = time.time()

            detections = detector.predict_gpu(
                frame,
                conf_threshold=CONF_THRESHOLD,
            )

            inference_time = time.time() - inference_start

            # ------------------------------------------------
            # DRAW DETECTIONS
            # ------------------------------------------------

            frame = draw_detections(
                frame,
                detections,
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.time()

            elapsed = current_time - prev_time

            fps = 1.0 / max(
                elapsed,
                1e-6,
            )

            prev_time = current_time

            # ------------------------------------------------
            # FPS TEXT
            # ------------------------------------------------

            cv2.putText(
                frame,
                "FPS: {:.1f}".format(fps),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # INFERENCE TIME
            # ------------------------------------------------

            cv2.putText(
                frame,
                "Inference: {:.1f} ms".format(
                    inference_time * 1000.0
                ),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # NUMBER OF DETECTIONS
            # ------------------------------------------------

            cv2.putText(
                frame,
                "Objects: {}".format(
                    len(detections)
                ),
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # DISPLAY FRAME
            # ------------------------------------------------

            cv2.imshow(
                window_name,
                frame,
            )

            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # q
            if key == ord("q"):
                print("[INFO] Pressed q.")
                break

            # ESC
            if key == 27:
                print("[INFO] Pressed ESC.")
                break

            # ------------------------------------------------
            # WINDOW CLOSE BUTTON
            # ------------------------------------------------

            try:

                if cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                ) < 1:

                    print("[INFO] Window closed.")
                    break

            except cv2.error:

                break

    except KeyboardInterrupt:

        print("\n[INFO] Interrupted by user.")

    except Exception as e:

        print("\n" + "=" * 60)
        print("[ERROR] Vision test failed")
        print("=" * 60)

        print(str(e))

        raise

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        print("[INFO] Cleaning up...")

        if camera is not None:

            try:
                camera.stop()
            except Exception as e:
                print(
                    "[WARNING] Camera stop error:",
                    e,
                )

        cv2.destroyAllWindows()

        print("[INFO] Test finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
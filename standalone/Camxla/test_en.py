# chay doc lap khong lien quan den UART
import argparse
import time
import cv2
import yaml

# Import cac class Model co san trong source code cua cau
# (Thay doi ten module/class ben duoi cho dung voi du an cua cau)
# from models.v1_detector import SpearHeadDetector
# from models.v2_detector import KFSDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Vision Test")
    parser.add_argument(
        "--config",
        type=str,
        default="config_test_v1.yaml",
        help="Path to test config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Nap file cau hinh test
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Da nap cau hinh: {args.config}")
    print(f"[INFO] Che do nap model (load_mode): {cfg['system']['load_mode']}")

    # 2. Khoi tao Camera hoac Anh tinh
    use_static_img = cfg["system"].get("test_image", False)
    if not use_static_img:
        cap = cv2.VideoCapture(cfg["hardware"]["camera"]["device_id"])
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, cfg["hardware"]["camera"]["width"]
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, cfg["hardware"]["camera"]["height"]
        )
        if not cap.isOpened():
            print("[ERROR] Khong the mo Camera!")
            return

    # 3. Khoi tao Model tuong ung theo load_mode
    load_mode = cfg["system"]["load_mode"]
    detector = None

    if load_mode == 2:
        print("[INFO] Dang khoi tao mo hinh V1 (SpearHead)...")
        # detector = SpearHeadDetector(cfg['v1_model'])
    elif load_mode == 1:
        print("[INFO] Dang khoi tao mo hinh V2 (KFS Cascade)...")
        # detector = KFSDetector(cfg['v2_model'])

    window_name = f"Standalone Test - Mode {load_mode}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()

    # 4. Luong xu ly va hien thi chinh
    while True:
        if use_static_img:
            frame = cv2.imread(cfg["system"]["test_image_path"])
            if frame is None:
                print("[ERROR] Khong tim thay anh test tinh!")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Mat luong camera...")
                break

        # --- CHAY SUY LUAN MO HINH ---
        # (Goi ham predict/detect tu class model cua cau)
        # results, roi_crops = detector.detect(frame)

        # Tinh toan FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # --- VE THONG TIN LEN FRAME ---
        # Ve FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Gia dinh ve mau Bounding Box de minh hoa
        # For box in results:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #     cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), ...)

        # Hien thi khung hinh chinh
        cv2.imshow(window_name, frame)

        # DAC BIET DANH CHO V2: Quan sat truc tiep anh Crop ROI 64x64 dua vao CNN
        # if load_mode == 1 and roi_crops:
        #     # Hien thi vung anh cat vua phong lon de soi xem CNN nhin thay gi
        #     debug_roi = cv2.resize(roi_crops[0], (200, 200))
        #     cv2.imshow("V2 CNN Crop Input Debug", debug_roi)

        # Nhan phim 'q' hoac 'ESC' de thoat
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Don dep tai nguyen
    if not use_static_img:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Da thoat chuong trinh test thanh cong.")


if __name__ == "__main__":
    main()
# chay doc lap khong lien quan den UART
import argparse
import sys
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


def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Khong tim thay file cau hinh: {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] File cau hinh YAML loi: {e}")
        sys.exit(1)

    if not isinstance(cfg, dict) or "system" not in cfg:
        print("[ERROR] Cau hinh thieu muc 'system'!")
        sys.exit(1)
    return cfg


def open_camera(cam_cfg):
    cap = cv2.VideoCapture(cam_cfg.get("device_id", 0))
    # Kiem tra mo camera TRUOC khi set thong so
    if not cap.isOpened():
        cap.release()
        return None
    if "width" in cam_cfg:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    if "height" in cam_cfg:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    return cap


def build_detector(load_mode, cfg):
    if load_mode == 2:
        print("[INFO] Dang khoi tao mo hinh V1 (SpearHead)...")
        # return SpearHeadDetector(cfg['v1_model'])
    elif load_mode == 1:
        print("[INFO] Dang khoi tao mo hinh V2 (KFS Cascade)...")
        # return KFSDetector(cfg['v2_model'])
    else:
        print(f"[ERROR] load_mode khong hop le: {load_mode} (chi ho tro 1 hoac 2)")
        sys.exit(1)
    return None  # Placeholder cho den khi bo comment cac dong tren


def main():
    args = parse_args()

    # 1. Nap file cau hinh test
    cfg = load_config(args.config)
    load_mode = cfg["system"].get("load_mode")

    print(f"[INFO] Da nap cau hinh: {args.config}")
    print(f"[INFO] Che do nap model (load_mode): {load_mode}")

    # 2. Khoi tao Camera hoac Anh tinh
    use_static_img = cfg["system"].get("test_image", False)
    cap = None
    static_frame = None

    if use_static_img:
        # Doc anh MOT LAN duy nhat, khong doc lai moi vong lap
        static_frame = cv2.imread(cfg["system"]["test_image_path"])
        if static_frame is None:
            print("[ERROR] Khong tim thay anh test tinh!")
            return
    else:
        cap = open_camera(cfg["hardware"]["camera"])
        if cap is None:
            print("[ERROR] Khong the mo Camera!")
            return

    try:
        # 3. Khoi tao Model tuong ung theo load_mode
        detector = build_detector(load_mode, cfg)

        window_name = f"Standalone Test - Mode {load_mode}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = time.time()

        # 4. Luong xu ly va hien thi chinh
        while True:
            if use_static_img:
                # copy de ve len khong lam ban anh goc
                frame = static_frame.copy()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Mat luong camera...")
                    break

            # --- CHAY SUY LUAN MO HINH ---
            results, roi_crops = [], []
            if detector is not None:
                # (Goi ham predict/detect tu class model cua cau)
                results, roi_crops = detector.detect(frame)

            # Tinh toan FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # --- VE THONG TIN LEN FRAME ---
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
            # for (x1, y1, x2, y2, label, conf) in results:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #     cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Hien thi khung hinh chinh
            cv2.imshow(window_name, frame)

            # DAC BIET DANH CHO V2: Quan sat truc tiep anh Crop ROI 64x64 dua vao CNN
            if load_mode == 1 and roi_crops:
                # Hien thi vung anh cat vua phong lon de soi xem CNN nhin thay gi
                debug_roi = cv2.resize(roi_crops[0], (200, 200))
                cv2.imshow("V2 CNN Crop Input Debug", debug_roi)

            # Nhan phim 'q' hoac 'ESC' de thoat
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            # Dong cua so bang nut X
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Don dep tai nguyen (chay ca khi co exception / Ctrl+C)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    print("[INFO] Da thoat chuong trinh test.")


if __name__ == "__main__":
    main()
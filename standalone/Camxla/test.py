#chay doc lap khong lien quan den UART
import argparse
import time
import cv2
import yaml

# Import các class Model có sẵn trong source code của cậu
# (Thay đổi tên module/class bên dưới cho đúng với dự án của cậu)
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

    # 1. Nạp file cấu hình test
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Đã nạp cấu hình: {args.config}")
    print(f"[INFO] Chế độ nạp model (load_mode): {cfg['system']['load_mode']}")

    # 2. Khởi tạo Camera hoặc Ảnh tĩnh
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
            print("[ERROR] Không thể mở Camera!")
            return

    # 3. Khởi tạo Model tương ứng theo load_mode
    load_mode = cfg["system"]["load_mode"]
    detector = None

    if load_mode == 2:
        print("[INFO] Đang khởi tạo mô hình V1 (SpearHead)...")
        # detector = SpearHeadDetector(cfg['v1_model'])
    elif load_mode == 1:
        print("[INFO] Đang khởi tạo mô hình V2 (KFS Cascade)...")
        # detector = KFSDetector(cfg['v2_model'])

    window_name = f"Standalone Test - Mode {load_mode}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()

    # 4. Luồng xử lý và hiển thị chính
    while True:
        if use_static_img:
            frame = cv2.imread(cfg["system"]["test_image_path"])
            if frame is None:
                print("[ERROR] Không tìm thấy ảnh test tĩnh!")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Mất luồng camera...")
                break

        # --- CHẠY SUY LUẬN MÔ HÌNH ---
        # (Gọi hàm predict/detect từ class model của cậu)
        # results, roi_crops = detector.detect(frame)

        # Tính toán FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # --- VẼ THÔNG TIN LÊN FRAME ---
        # Vẽ FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Giả định vẽ mẫu Bounding Box để minh họa
        # For box in results:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #     cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), ...)

        # Hiển thị khung hình chính
        cv2.imshow(window_name, frame)

        # ĐẶC BIỆT DÀNH CHO V2: Quan sát trực tiếp ảnh Crop ROI 64x64 đưa vào CNN
        # if load_mode == 1 and roi_crops:
        #     # Hiển thị vùng ảnh cắt vừa phóng lớn để soi xem CNN nhìn thấy gì
        #     debug_roi = cv2.resize(roi_crops[0], (200, 200))
        #     cv2.imshow("V2 CNN Crop Input Debug", debug_roi)

        # Nhấn phím 'q' hoặc 'ESC' để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Dọn dẹp tài nguyên
    if not use_static_img:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát chương trình test thành công.")


if __name__ == "__main__":
    main()
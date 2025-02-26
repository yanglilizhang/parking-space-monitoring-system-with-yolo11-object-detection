from ultralytics import YOLO
from src.psms_lib import psms
import cv2

yolo_model = YOLO('src/model/yolo11n.pt')
class_names = yolo_model.names
zones, number_of_zones, frame_width, frame_height = psms.extract_data_from_file("src/data_file/data.txt")
cap = psms.load_camera("inference/demo.mp4")

success = True
frame_name = "PSMS"
wait_key = 1
ord_key = 'q'

while success:
    success, frame = cap.read()
    if not success:
        break

    # 对帧进行推理
    results = yolo_model(frame)

    # 打印模型的输出信息
    for result in results:
        print("Inference Results:", result)
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cls = int(box.cls[0])
            conf = box.conf[0]
            print(f"Class: {class_names[cls]}, Confidence: {conf:.2f}, Bounding Box: {r}")

    # 显示帧
    cv2.imshow(frame_name, frame)

    # 按 'q' 键退出
    if cv2.waitKey(wait_key) & 0xFF == ord(ord_key):
        break

cap.release()
cv2.destroyAllWindows()

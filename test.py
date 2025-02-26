from ultralytics import YOLO
from src.psms_lib import psms
import cv2

yolo_model = YOLO('src/model/yolo11n.pt')
class_names = yolo_model.names
# Model Classes: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
# 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
# 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
# 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
# 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
# 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
# 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
# 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
# 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
# 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
# 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
# 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
print("Model Classes:", class_names)
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
    # print("Inference Results:", results)

    # results2 = yolo_model.predict(source=[frame], save=False, conf=0.5)
    # print("Inference Results predict:", results2[0]) # 最外层

    # 打印模型的输出信息
    for result in results:
        # print('----------------------------------->')
        # print("Inference Results:", result)
        # print('----------------------------------->')
        # print("Model Output boxes:", result.boxes)
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # print("Box:", box)
            # cls: array([2], dtype=float32)
            # conf: array([0.56985], dtype=float32)
            # data: array([[739.74, 694.44, 873.13, 825.79, 0.56985, 2]], dtype=float32)
            # id: None
            # is_track: False
            # orig_shape: (1080, 1920)
            # shape: (1, 6)
            # xywh: array([[806.43, 760.12, 133.39, 131.35]], dtype=float32)
            # xyxy: array([[739.74, 694.44, 873.13, 825.79]], dtype=float32)
            # Class: car, Confidence: 0.57, Bounding Box: [739 694 873 825]
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

from ultralytics import YOLO
from src.psms_lib import psms
import cv2


def main():
    # 实例化YOLO模型，用于后续的目标检测
    yolo_model = YOLO('src/model/yolo11n.pt')
    class_names = yolo_model.names
    # 从指定的数据文件中提取区域和帧信息
    zones, number_of_zones, frame_width, frame_height = psms.extract_data_from_file("src/data_file/data.txt")
    cap = psms.load_camera("inference/demo.mp4")
    success = True
    frame_name = "PSMS"
    wait_key = 1
    ord_key = 'q'

    while success:
        ret, frame = cap.read()

        if not ret:
            break

        # 初始化区域列表 如果 number_of_zones 为3，则返回 [[], [], []]
        zones_list = psms.init_zone_list(number_of_zones)
        frame = cv2.resize(frame, (frame_width, frame_height))
        # 获取YOLO模型的预测边界框
        boxes = psms.get_prediction_boxes(frame, yolo_model, 0.15)
        print("boxes:", boxes)
        # 可选：绘制区域多边形线条
        psms.draw_polylines_zones(frame, zones)  # Optional
        # 跟踪区域内的物体并更新区域状态
        frame, zones_list = psms.track_objects_in_zones(frame, boxes, zones, zones_list, class_names)
        # [[], [], [], [], [], [], [], [], [], [], ['car'], []]
        print("zones_list:", zones_list)
        # 统计被占用的区域数量
        numb_of_occupied = psms.count_occupied_space(number_of_zones, zones_list)

        data_to_display = {
            "numb_of_occupied": numb_of_occupied,
            "number_of_zones": number_of_zones
        }
        # 可选：显示区域信息
        psms.display_zone_info(frame, data_to_display)  # Optional
        # 可选：显示当前帧并等待按键输入
        success = psms.show_frame(frame, frame_name, wait_key, ord_key)  # Optional

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

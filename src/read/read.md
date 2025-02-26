
results = yolo_model(frame)
for result in results:
    print(f"Class: {result.cls}, Confidence: {result.conf}, Bounding Box: {result.xyxy[0].tolist()}")
    print("Inference Results:", result)
    print('----------------------------------->')
    print("Model Output names:", result.names)
    print("Model Output boxes:", result.boxes)
    print("Model Output boxes.data:", result.boxes.data)


----------------------------------->
Inference Results: ultralytics.engine.results.Results object with attributes:

>>boxes: ultralytics.engine.results.Boxes object
>>keypoints: None
>>masks: None
>>names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
>>obb: None
>>orig_img: array([[[ 69,  42,  47],
        [ 87,  60,  65],
        [102,  70,  76],
        ...,
        [ 90,  79,  76],
        [ 74,  67,  64],
        [ 57,  50,  47]],  
        ...,
        [ 47,  38,  42],
        [ 43,  34,  38],
        [ 35,  26,  30]]], dtype=uint8)
>>orig_shape: (1080, 1920)
>>path: 'image0.jpg'
>>probs: None
>>save_dir: 'runs\\detect\\predict'
>>speed: {'preprocess': 2.5989000000001816, 'inference': 70.85140000000001, 'postprocess': 0.7830000000002002}
----------------------------------->
Model Output names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

Model Output boxes: ultralytics.engine.results.Boxes object with attributes:
>>cls: tensor([7., 7., 2., 7., 2., 2.])
>>conf: tensor([0.4240, 0.4231, 0.3517, 0.3273, 0.3116, 0.2863])
>>data: tensor([[8.6560e+02, 5.6996e+02, 9.9929e+02, 6.4054e+02, 4.2403e-01, 7.0000e+00],
        [1.4524e+03, 4.7699e+02, 1.6336e+03, 5.8096e+02, 4.2314e-01, 7.0000e+00],
        [8.6476e+02, 5.6920e+02, 9.9900e+02, 6.4101e+02, 3.5171e-01, 2.0000e+00],
        [1.3570e+03, 3.7603e+02, 1.4626e+03, 4.7068e+02, 3.2728e-01, 7.0000e+00],
        [1.4371e+03, 4.0286e+02, 1.4948e+03, 4.5217e+02, 3.1163e-01, 2.0000e+00],
        [1.2195e+02, 5.8740e+02, 1.7395e+02, 6.5094e+02, 2.8634e-01, 2.0000e+00]])
>>id: None
>>is_track: False
>>orig_shape: (1080, 1920)
>>shape: torch.Size([6, 6])
>>xywh: tensor([[ 932.4476,  605.2501,  133.6937,   70.5771],
        [1542.9631,  528.9781,  181.2233,  103.9738],
        [ 931.8799,  605.1046,  134.2322,   71.8165],
        [1409.8240,  423.3557,  105.6438,   94.6496],
        [1465.9060,  427.5123,   57.7094,   49.3115],
        [ 147.9493,  619.1707,   51.9973,   63.5360]])
>>xywhn: tensor([[0.4856, 0.5604, 0.0696, 0.0653],
        [0.8036, 0.4898, 0.0944, 0.0963],
        [0.4854, 0.5603, 0.0699, 0.0665],
        [0.7343, 0.3920, 0.0550, 0.0876],
        [0.7635, 0.3958, 0.0301, 0.0457],
        [0.0771, 0.5733, 0.0271, 0.0588]])
>>xyxy: tensor([[ 865.6007,  569.9616,  999.2944,  640.5387],
        [1452.3514,  476.9912, 1633.5747,  580.9650],
        [ 864.7637,  569.1964,  998.9960,  641.0129],
        [1357.0021,  376.0309, 1462.6459,  470.6805],
        [1437.0514,  402.8566, 1494.7607,  452.1681],
        [ 121.9506,  587.4027,  173.9479,  650.9387]])
>>xyxyn: tensor([[0.4508, 0.5277, 0.5205, 0.5931],
        [0.7564, 0.4417, 0.8508, 0.5379],
        [0.4504, 0.5270, 0.5203, 0.5935],
        [0.7068, 0.3482, 0.7618, 0.4358],
        [0.7485, 0.3730, 0.7785, 0.4187],
        [0.0635, 0.5439, 0.0906, 0.6027]]) 

Class: truck, Confidence: 0.42, Bounding Box: [865 569 999 640]
Class: truck, Confidence: 0.42, Bounding Box: [1452  476 1633  580]
Class: car, Confidence: 0.35, Bounding Box: [864 569 998 641]
Class: truck, Confidence: 0.33, Bounding Box: [1357  376 1462  470]
Class: car, Confidence: 0.31, Bounding Box: [1437  402 1494  452]
Class: car, Confidence: 0.29, Bounding Box: [121 587 173 650]

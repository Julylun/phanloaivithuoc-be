from snap7.util import *
from ultralytics import YOLO
import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ai", "best.pt")

model = YOLO(MODEL_PATH)

def detect(frame):
    result = model(frame, verbose=False)
    
    _result = result[0]
    return _result



def draw_yolo_boxes(result):
    """
    Vẽ bounding boxes và nhãn từ kết quả YOLOv8 lên ảnh.
    """
    # Lấy ảnh gốc (numpy array BGR)
    img = result.orig_img.copy()
    
    # Nếu không có box nào thì trả về ảnh gốc
    if result.boxes is None:
        return img

    # Lấy danh sách box, class, confidence
    boxes = result.boxes.xyxy.cpu().numpy()  # to numpy [x1, y1, x2, y2]
    confs = result.boxes.conf.cpu().numpy()  # confidence
    classes = result.boxes.cls.cpu().numpy().astype(int)  # class index

    for (box, cls, conf) in zip(boxes, classes, confs):
        x1, y1, x2, y2 = box.astype(int)
        label = result.names.get(cls, str(cls))
        color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)  # màu xanh nếu >0.5, đỏ nếu thấp hơn

        # Vẽ box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Vẽ label + confidence
        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img
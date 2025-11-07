import cv2
import os
import threading
from ultralytics import YOLO
from snap7.util import *
from queue import Queue
import time
import ffmpeg
import io
from typing import Iterator


#Thu muc goc -> app
# BASE_DIR = os.path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Di chuyen tu BaseDir vao best.pt
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'ai', 'best.pt')


class YoloService:
    def __init__(self):
        _max_queue = 10
        print('---')
        print('Khởi tạo Yolo với các thông số sau:')
        print('- Đường dẫn model: ', MODEL_PATH)
        print('- Camera: 0')
        print('- Hàng chờ tối đa, max queue: ', _max_queue)
        print('---')
        self.model = YOLO(MODEL_PATH)
        self.cap = cv2.VideoCapture(0)
        self.frame_queue = Queue(maxsize=_max_queue)

        self.running = False
        pass 

    def start_detection(self):
        """Bắt đầu nhận điện."""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.start()
    
    def stop_detection(self):
        """Dừng nhận điện."""
        self.running = False
        if self.cap:
            self.cap.release()
        pass

    def _detection_loop(self):
        """Hàm xử lý quá trình nhận diện"""
        while(self.running):
            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception('Can not grab frame from camera')

                #Đưa ảnh vào Model và lấy kết quả nhận diện
                results = self.model(frame, verbose=False)
                annotated_frame = results[0].plot() #Vẽ box lên

                # time.sleep(3)

                detection_data = {
                    'frame': annotated_frame,
                    'detection': results[0].boxes.data.tolist() if results[0].boxes else []
                }
            
                #Đưa kết quả vào một hàng đợi, nếu hàng đợi đầy thì khỏi
                if not self.frame_queue.full():
                    self.frame_queue.put(detection_data)

                #Giới hạn tốc độ xử lý 33 ~ 30fps
                cv2.waitKey(33)
            except Exception as e:
                print('Try to detect after 3s...')
                time.sleep(3)

    @staticmethod
    def draw_boxes(result):
        """Vẽ bounding boxes và nhãn từ kết quả YOLOv8 lên ảnh."""

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
    
    def get_latest_frame(self):
        """Lấy frame mới nhất từ queue để gửi sang FE."""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
yolo_service = YoloService()
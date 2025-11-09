from enum import Enum
import cv2
import os
import threading
from ultralytics import YOLO
from snap7.util import *
from queue import Queue
import time
import ffmpeg
import io
from typing import Counter, Iterator
from queue import Queue

CLASS_MAP = {
    0: "Normal",
    1: "Broken",
    2: "Missing_bill",
    3: "Missing_corner"
}


def show_yolo_image(img_array, title="YOLO Output"):
    if img_array is None:
        print("Ảnh không hợp lệ.")
        return
    # YOLO / OpenCV dùng định dạng BGR → hiển thị bằng cv2 trực tiếp
    cv2.imshow(title, img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Thu muc goc -> app
# BASE_DIR = os.path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Di chuyen tu BaseDir vao best.pt
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'ai', 'best.pt')


class Pill_Class(Enum):
    NORMAL = 0
    BROKEN = 1
    MISSING_PILL = 2
    MISSING_CORNER = 3
    BROKEN_AND_MISSING_CORNER = 4
    MISSING_PILL_AND_BROKEN = 5
    
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
        self.retry_times = 5

        self.running = False

        self.last_result = {"class_numer": -1, "detected_image": None}
        pass 

    def start_detection(self):
        """Bắt đầu nhận điện."""
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.start()
    
    def stop_detection(self):
        """Dừng nhận điện."""
        self.running = False
        if self.cap:
            self.cap.release()
        pass

    @staticmethod
    def _detect_class(class_list) -> Pill_Class:
        """
        Missing Pill & Broken > Missing Pill > Broken & Missing Corner > Boken || Missing corner > Normal
        """
        if Pill_Class.MISSING_PILL.value in class_list and Pill_Class.BROKEN.value in class_list:
            return Pill_Class.MISSING_PILL_AND_BROKEN
        if Pill_Class.MISSING_PILL.value in class_list:
            return Pill_Class.MISSING_PILL
        if Pill_Class.BROKEN.value in class_list and Pill_Class.MISSING_CORNER.value in class_list:
            return Pill_Class.BROKEN_AND_MISSING_CORNER
        if Pill_Class.BROKEN.value in class_list:
            return Pill_Class.BROKEN
        if Pill_Class.MISSING_CORNER.value in class_list:
            return Pill_Class.MISSING_CORNER
        return Pill_Class.NORMAL
    

    def tinhTong(a, b):
        return a + b
    


    def _detection_loop(self):
        """Hàm xử lý quá trình nhận diện"""
        collecting = False
        temp_result_queue = Queue()
        stop_counting_value = self.retry_times
        while(self.running):
            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception('Can not grab frame from camera')

                #Đưa ảnh vào Model và lấy kết quả nhận diện
                results = self.model(frame, verbose=False)
                annotated_frame = results[0].plot() #Vẽ box lê
                result = results[0]

                #Nếu đang thu thập + số lượng thuộc được nhận diện = 5 thì bắt đầu quá trình thu thập
                if collecting is not True and hasattr(result, 'boxes') and len(result.boxes) == 5:
                    collecting = True

                #Nếu đang thu thập + số lượng thuốc = 0 thì kết thúc thu thập và đưa ra kết quả
                if collecting and hasattr(result, 'boxes') and len(result.boxes) == 0:
                    if stop_counting_value > 0:
                        stop_counting_value-=1
                    else:
                        collecting = False 
                        result_classes = []
                        result_list = []

                        while not temp_result_queue.empty():
                            result = temp_result_queue.get()
                            result_list.append(result)

                            class_lists = []
                            for box in result.boxes: #Lấy class trong vỉ thuốc
                                cls = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
                                class_lists.append(cls)

                            #Đưa loại vỉ thuốc vào mảng result_classes
                            result_classes.append(YoloService._detect_class(class_lists))
                    
                        #Đếm số class có số lượng cao nhất để làm kết quả cuối cùng
                        counted_result = Counter(result_classes)
                        highest_class, highest_count = counted_result.most_common(1)[0]

                        #Lấy ảnh kết quả ở giữa
                        image_index = int(len(result_list) / 2)
                        while True:
                            _result = result_list[image_index]
                            class_lists = []
                            for box in result.boxes: #Lấy class trong vỉ thuốc
                                cls = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
                                class_lists.append(cls)

                            if YoloService._detect_class(class_lists) == highest_class:
                                break

                            if image_index >= len(result_list) - 1:
                                image_index = 0

                            image_index+=1

                        last_result = {
                            "class_number": highest_class.value,
                            "detected_image": YoloService.draw_boxes(result_list[image_index])
                        }
                        image_text = str(image_index) + '/' + str(len(result_list)) 
                        print(image_index)
                        print(last_result)
                        show_yolo_image(YoloService.draw_boxes(result_list[image_index]))
                        #TODO: send last result to front end
                    


                #Đang thu thập + số thuốc = 5 thì vào Queue tạm thời
                if collecting and hasattr(result, 'boxes') and len(result.boxes) == 5:
                    stop_counting_value = self.retry_times
                    temp_result_queue.put(result)


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
                print(e)
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
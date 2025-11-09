import asyncio
import base64
import json
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

import time

from fastapi.responses import StreamingResponse
from app.services.yolo_service import YoloService,yolo_service
from .webrtc_routes import webrtc_router
import cv2


# Khởi tạo router
router = APIRouter()

# Thêm routes vào router
# router.include_router(webrtc_router)
print('included')

@router.get("/")
async def home():
    return {"message": "Welcome to Phanloaivithuoc"}

@router.post("/detection/start")
async def start_detection():
    yolo_service.start_detection()

@router.post("/detection/stop")
async def stop_detection():
    yolo_service.stop_detection()

@router.post("/detection/export")
async def export_detection():
    print('exported!!!')




@router.websocket("/ws/video-stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("INFO: connection open")  # Log như của bạn
    try:
        while True:
            frame_data = yolo_service.get_latest_frame()  # {'frame': cv_frame, 'detection': [...]}
            if frame_data and frame_data['frame'] is not None:
                frame = frame_data['frame'].copy()
                
                # Fix: Parse detections đúng format YOLO (list of [x1,y1,x2,y2,conf,class_id])
                detections = frame_data.get('detection', [])
                num_dets = 0
                if detections:  # Nếu có detections (không phải empty list)
                    try:
                        for det in detections:
                            if isinstance(det, (list, tuple)) and len(det) >= 5:  # Format [x1,y1,x2,y2,conf,class_id,...]
                                x1, y1, x2, y2 = map(int, det[:4])  # Lấy bbox từ 4 phần tử đầu
                                conf = det[4] if len(det) > 4 else 0.0
                                class_id = int(det[5]) if len(det) > 5 else 0
                                label = f"Class {class_id} {conf:.2f}"  # Hoặc map class_id sang name (e.g., COCO labels)
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                num_dets += 1
                            else:
                                print(f"Warning: Invalid det format: {det}")  # Debug
                    except Exception as parse_err:
                        print(f"Detections parse error: {parse_err}")
                        num_dets = 0  # Skip vẽ nếu lỗi
                
                # Resize
                frame = cv2.resize(frame, (640, 480))
                
                # Encode base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Payload (gửi detections raw để client dùng nếu cần)
                payload = {
                    'image': f"data:image/jpeg;base64,{img_base64}",
                    'detections': detections  # Raw list để client parse nếu muốn overlay JS
                }
                await websocket.send_text(json.dumps(payload))
                # print(f"Sent frame with {num_dets} detections")  # Sửa log dùng num_dets
                
            await asyncio.sleep(0.033)  # ~30 FPS
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Stream error: {e}")
        # Không break, tiếp tục loop nếu có thể
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

@router.get("/detection/get-result")
async def get_detection_result():
    print(yolo_service.running)
    return yolo_service.last_result


@router.websocket("/ws/video-stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("INFO: connection open")  # Log như của bạn
    while True:
        try:
            frame_data = yolo_service.get_latest_frame()  # {'frame': cv_frame, 'detection': [...]}
            if frame_data and frame_data['frame'] is not None:
                frame = frame_data['original_frame'].copy()
               

                frame = cv2.resize(frame, (640, 480))
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    'image': f"data:image/jpeg;base64,{img_base64}",
                    'detections': frame_data['detection']
                }
                await websocket.send_text(json.dumps(payload))
                
        except WebSocketDisconnect:
            print("Client disconnected")
            break
        except Exception as e:
            print(f"Stream error: {e}")
        # Không break, tiếp tục loop nếu có thể
        await asyncio.sleep(0.033)  # ~30 FPS
    
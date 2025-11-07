import time
from fastapi import APIRouter, Request
import json
from aiortc import MediaStreamError, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import asyncio, json
import cv2
from fastrtc import Stream
import numpy as np
from snap7.util import *

from app.services import yolo_service
from ..services.model import detect, draw_yolo_boxes
from fastapi import FastAPI


webrtc_router = APIRouter(
    prefix="/webrtc",
    tags=["webrtc"]
)

pcs = set()

@webrtc_router.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # 🔹 THÊM TRANSCEIVER TRƯỚC
    # (đảm bảo aiortc có media section tương ứng)
    pc.addTransceiver("video", direction="recvonly")
    # pc.addTransceiver("audio", direction="recvonly")

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        print(f"📸 Received track: {track.kind}")
        if track.kind == "video":
            frame_count = 0
            target_fps = 15  # Giới hạn FPS
            frame_interval = 1.0 / target_fps
            last_processed = 0
            is_processing = False  # Flag theo dõi trạng thái xử lý YOLO

            cap = cv2.VideoCapture(0)

            while True:
                try:
                    ret, frame = cap.read()


                    print(ret)
                    if not ret:
                        print("[Camera] Lỗi đọc khung hình...")
                        time.sleep(0.2)
                        continue


                    # # Nhận frame với timeout để tránh treo
                    # frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                    current_time = asyncio.get_event_loop().time()

                    # Chỉ xử lý nếu đủ thời gian và không đang chạy YOLO
                    if current_time - last_processed >= frame_interval and not is_processing:
                        is_processing = True  # Đánh dấu đang xử lý
                        np_frame = frame.to_ndarray(format="bgr24")
                        print(f"Frame resolution: {np_frame.shape}")

                        try:
                            results = detect(frame=np_frame)
                            _frame = draw_yolo_boxes(result=results)
                            cv2.imshow("YOLOv8 Detection", _frame)
                        except Exception as e:
                            print(f"Error in YOLO detection: {str(e)}")
                        finally:
                            is_processing = False  # Kết thúc xử lý
                            last_processed = current_time
                            frame_count += 1

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except asyncio.TimeoutError:
                    print("Timeout waiting for frame")
                    continue
                except MediaStreamError:
                    print("MediaStreamError: Track has been stopped or ended")
                    break
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    break 


    # 🔹 Đặt mô tả của client
    await pc.setRemoteDescription(offer)

    # 🔹 Tạo answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }



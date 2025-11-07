from contextlib import asynccontextmanager
from fastapi import FastAPI
from .routes.router import router
from fastapi.middleware.cors import CORSMiddleware
from .services.yolo_service import yolo_service


# Integrate vào FastAPI lifespan của bạn
@asynccontextmanager
async def lifespan(app: FastAPI):
    yolo_service.start_detection()
    yield
    yolo_service.stop_detection()

app = FastAPI(title="Phân loại vỉ thuốc", lifespan=lifespan)

# ✅ Cấu hình CORS
origins = [
    "http://localhost:3000",  # FE (Next.js)
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ['http://localhost:3000'] nếu FE chạy port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gắn route vào app chính
app.include_router(router=router)
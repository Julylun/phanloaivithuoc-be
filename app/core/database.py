from motor.motor_asyncio import AsyncIOMotorClient

# Kết nối tới cơ sở dữ liệu
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["phanloaivithuoc"]
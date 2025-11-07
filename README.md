# Cách chạy
- Tải uv
- Chạy lệnh uv sync
- Source vào .venv
- Chạy:
```
uvicorn app.main:app --reload
```

# Cấu trúc thư mục

```
Phanloaivithuoc-be/
│
├── app/
│   ├── main.py                # Điểm bắt đầu (entry point)
│   ├── core/
│   │   ├── config.py          # Cấu hình (Mongo URI, secret key,...)
│   │   └── database.py        # Kết nối MongoDB
│   │
│   ├── models/
│   │  
│   │
│   ├── routes/
│   │   └── router.py     # Add các endpoint
│   │
│   ├── services/ #Xử lý logic code
│   │   
│   │
│   └── __init__.py
│
├── requirements.txt           # Danh sách thư viện
└── README.md
```
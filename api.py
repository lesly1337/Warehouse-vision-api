import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import cv2
import numpy as np
from database import init_db, async_sessionmaker, ProcessingHistory
from contextlib import asynccontextmanager
import torch

class BoxDetails(BaseModel):
    confidence: float
    coordinates: List[int]

class ImagePredictResponse(BaseModel):
    status: str
    filename: str
    total_boxes_found: int
    details: List[BoxDetails]

class VideoPredictResponse(BaseModel):
    status: str
    filename: str
    total_boxes_counted: int
    unique_ids: List[int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Инициализация базы данных...")
    await init_db()
    yield
    print("Сервер Warehouse Vision остановлен.")

app = FastAPI(title="Warehouse Vision API", lifespan=lifespan)


print("Загрузка весов модели...")
model = YOLO('best.pt') 
print("Модель готова к работе!")


@app.get("/")
async def root():
    return {"message": "Система Warehouse Vision онлайн. Перейдите на /docs для тестирования."}

# Анализ одной картинки
@app.post("/api/v1/detect-boxes", response_model=ImagePredictResponse)
async def detect_boxes(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(frame, conf=0.5, verbose=False)
    
    detected_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detected_boxes.append({
                "confidence": round(conf, 2),
                "coordinates": [int(x1), int(y1), int(x2), int(y2)]
            })
            
    return {
        "status": "success",
        "filename": file.filename,
        "total_boxes_found": len(detected_boxes),
        "details": detected_boxes
    }

# Анализ целого видео
@app.post("/api/v1/analyze-video", response_model=VideoPredictResponse)
async def analyze_video(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())
        
    cap = cv2.VideoCapture(temp_filename)
    counted_ids = set()
    total_boxes = 0
    line_x = 600

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_infer = frame.copy()
        frame_infer[0:int(h * 0.4), 0:w] = (0, 0, 0)

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        results = model.track(frame_infer, verbose=False, device="cpu", conf=0.5, persist=True, tracker="bytetrack.yaml")

        for r in results:
            boxes = r.boxes
            if boxes.id is None:
                continue
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                track_id = int(boxes.id[i])

                if cx > line_x and track_id not in counted_ids:
                    total_boxes += 1
                    counted_ids.add(track_id)

    cap.release()
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    async with async_sessionmaker() as session:
        new_record = ProcessingHistory(
            filename=file.filename,
            total_boxes=total_boxes
        )
        session.add(new_record)
        await session.commit()

    return {
        "status": "success",
        "filename": file.filename,
        "total_boxes_counted": total_boxes,
        "unique_ids": list(counted_ids)
    }
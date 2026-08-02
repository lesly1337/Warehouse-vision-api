from ultralytics import YOLO
import os

def main():
    model = YOLO('yolov8n.pt') 

    root = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(root, 'data.yaml')

    print("Начало дообучения модели на кастомном датасете...")
    
    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        device=0,
        workers=2
    )

    print("Обучение завершено!")

if __name__ == '__main__':
    main()
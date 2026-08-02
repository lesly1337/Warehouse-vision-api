import cv2 as cv
from ultralytics import YOLO
import os
import time

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    vidPth = os.path.join(root, 'conveyor.mp4')

    cap = cv.VideoCapture(vidPth)
    model = YOLO('best.pt')

    prev_time = 0
    counted_ids = set()
    total_boxes = 0
    line_x = 600
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или файл не найден.")
            break

        current_time = time.time()

        h, w = frame.shape[:2]

        frame_infer = frame.copy()

        frame_infer[0:int(h * 0.4), 0:w] = (0, 0, 0) 

        results = model.track(frame_infer, stream=True, verbose=False, device=0, conf=0.1, persist=True, tracker="bytetrack.yaml")

        for r in results:
            boxes = r.boxes

            if boxes.id is None:
                continue
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                cx = cx = int((x1 + x2) / 2)
                conf = int(box.conf[0] * 100)
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                track_id = int(boxes.id[i])

                if cx > line_x and track_id not in counted_ids:
                    total_boxes += 1
                    counted_ids.add(track_id)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                label = f"ID: {track_id} {class_name} {conf}%"
                cv.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv.line(frame, (600, 500), (600, 1500), (0, 255, 0), 3 )
        cv.putText(frame, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f"Total boxes: {int(total_boxes)}", (1020, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        cv.imshow('video1', frame)
        if cv.waitKey(int(1)) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
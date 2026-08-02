import cv2 as cv
import os
import sys

def extract_frames(video_name, step=10, subfolder_name=''):
    root = os.path.dirname(os.path.abspath(__file__))
    
    video_path = os.path.join(root, video_name)
    
    
    if subfolder_name:
        output_dir = os.path.join(root, 'dataset', subfolder_name)
    else:
        output_dir = os.path.join(root, 'dataset')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frame_count = 0
    saved_count = 0


    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        if frame_count % step == 0:
            img_name = f"img_{frame_count:04d}.png"
            img_path = os.path.join(output_dir, img_name)
            
            cv.imwrite(img_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Всего кадров в видео обработано: {frame_count}")
    print(f"Успешно сохранено в папку 'dataset': {saved_count} изображений.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
        
    video_file = sys.argv[1]
    
    target_subfolder = sys.argv[2] if len(sys.argv) > 2 else ''
        
    extract_frames(video_file, step=10, subfolder_name=target_subfolder)
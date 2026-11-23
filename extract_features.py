import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

LEFT = {'ear': 7, 'shoulder': 11, 'hip': 23, 'knee': 25, 'ankle': 27}
RIGHT = {'ear': 8, 'shoulder': 12, 'hip': 24, 'knee': 26, 'ankle': 28}

def compute_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dx, -dy) * 180 / np.pi

def choose_side(landmarks):
    def avg_vis(side):
        return np.mean([landmarks[idx].visibility for idx in side.values()])
    return LEFT if avg_vis(LEFT) >= avg_vis(RIGHT) else RIGHT

def extract_angles(landmarks, side):
    pts = {}
    for name, idx in side.items():
        pts[name] = (landmarks[idx].x, landmarks[idx].y)
    ear = pts['ear']
    shoulder = pts['shoulder']
    hip = pts['hip']
    knee = pts['knee']
    ankle = pts['ankle']
    return {
        'angle_ES': compute_angle(ear, shoulder),
        'angle_SH': compute_angle(shoulder, hip),
        'angle_HK': compute_angle(hip, knee),
        'angle_KA': compute_angle(knee, ankle)
    }

data_dir = 'posture_data'
good_dir = os.path.join(data_dir, 'good')
bad_dir = os.path.join(data_dir, 'bad')
records = []

def process_folder(folder_path, label, folder_name):
    print(f"Обработка папки: {folder_name}")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Найдено {len(files)} изображений в {folder_name}")
    for filename in files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Не удалось прочитать {filename}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            print(f"  Точки не найдены на {filename}")
            continue
        landmarks = results.pose_landmarks.landmark
        side = choose_side(landmarks)
        angles = extract_angles(landmarks, side)
        records.append({
            'filename': filename,
            'label': label,
            **angles
        })
        print(f"  Обработано: {filename}")

process_folder(good_dir, 1, 'good')
process_folder(bad_dir, 0, 'bad')

df = pd.DataFrame(records)
df.to_csv('posture_features.csv', index=False)
print(f"Сохранено {len(df)} записей в posture_features.csv")
import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

# Индексы точек (правая сторона, так как у вас фото в профиль)
RIGHT = {'ear': 8, 'shoulder': 12, 'hip': 24, 'knee': 26, 'ankle': 28}

print("Укажите путь к эталонному фото (хорошая поза):")
img_path = input("> ")

if not os.path.exists(img_path):
    print("Файл не найден!")
    exit()

img = cv2.imread(img_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)

if not results.pose_landmarks:
    print("Не удалось найти точки на изображении!")
    exit()

landmarks = results.pose_landmarks.landmark
# Сохраняем координаты (x, y) в нормализованном виде (от 0 до 1)
ref_points = {}
for name, idx in RIGHT.items():
    ref_points[name] = (landmarks[idx].x, landmarks[idx].y)

# Сохраняем в файл
np.save('reference_pose.npy', ref_points)
print("Эталонная поза сохранена в reference_pose.npy")
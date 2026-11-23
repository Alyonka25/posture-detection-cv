import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('posture_classifier.pkl')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT = {'ear': 7, 'shoulder': 11, 'hip': 23, 'knee': 25, 'ankle': 27}
RIGHT = {'ear': 8, 'shoulder': 12, 'hip': 24, 'knee': 26, 'ankle': 28}

def compute_angle(p1, p2):
    """Угол между вектором (p1->p2) и вертикалью (вверх) в градусах."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dx, -dy) * 180 / np.pi

def choose_side(landmarks):
    def avg_vis(side):
        return np.mean([landmarks[idx].visibility for idx in side.values()])
    return LEFT if avg_vis(LEFT) >= avg_vis(RIGHT) else RIGHT

def get_angles(landmarks, side):
    pts = {name: (landmarks[idx].x, landmarks[idx].y) for name, idx in side.items()}
    return {
        'ES': compute_angle(pts['ear'], pts['shoulder']),
        'SH': compute_angle(pts['shoulder'], pts['hip']),
        'HK': compute_angle(pts['hip'], pts['knee']),
        'KA': compute_angle(pts['knee'], pts['ankle'])
    }

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        side = choose_side(landmarks)
        angles = get_angles(landmarks, side)

        h, w = frame.shape[:2]
        pts_px = {}
        for name, idx in side.items():
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            pts_px[name] = (x, y)

        # ----- ЗЕЛЁНАЯ ВЕРТИКАЛЬ через лодыжку -----
        ankle_x = pts_px['ankle'][0]
        cv2.line(frame, (ankle_x, 0), (ankle_x, h), (0, 255, 0), 2)

        # ----- ТОЧКИ (жёлтые) -----
        for name, (x, y) in pts_px.items():
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(frame, name[:3], (x+8, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # ----- КЛАССИФИКАЦИЯ -----
        angles_list = [angles['ES'], angles['SH'], angles['HK'], angles['KA']]
        pred = model.predict([angles_list])[0]
        posture = "Good" if pred == 1 else "Bad"
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)

        # ----- ЕСЛИ BAD: РИСУЕМ КРАСНЫЙ СКЕЛЕТ И ПОДПИСЫВАЕМ УГЛЫ -----
        if pred == 0:
            order = ['ankle', 'knee', 'hip', 'shoulder', 'ear']
            # Соответствие сегментов и названий углов
            seg_angles = {
                ('ankle','knee'): 'KA',
                ('knee','hip'): 'HK',
                ('hip','shoulder'): 'SH',
                ('shoulder','ear'): 'ES'
            }
            for i in range(len(order)-1):
                pt1 = pts_px[order[i]]
                pt2 = pts_px[order[i+1]]
                cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
                # Подпись угла рядом с серединой отрезка
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                angle_name = seg_angles[(order[i], order[i+1])]
                angle_value = angles[angle_name]
                cv2.putText(frame, f"{angle_name}: {angle_value:.1f}°", (mid_x-20, mid_y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Общий угол отклонения (лодыжка–плечо)
            ankle_norm = (landmarks[side['ankle']].x, landmarks[side['ankle']].y)
            shoulder_norm = (landmarks[side['shoulder']].x, landmarks[side['shoulder']].y)
            dev_angle = compute_angle(ankle_norm, shoulder_norm)
            cv2.putText(frame, f"Overall deviation: {dev_angle:.1f} deg", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ----- ВСЕГДА ВЫВОДИМ СТАТУС И УГЛЫ (для информации) -----
        cv2.putText(frame, f"Posture: {posture}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"ES:{angles['ES']:.1f} SH:{angles['SH']:.1f} HK:{angles['HK']:.1f} KA:{angles['KA']:.1f}",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Posture Detection (Green = reference)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
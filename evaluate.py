import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Загрузка модели
model = joblib.load('posture_classifier.pkl')
print("Модель загружена.")

# 2. Загрузка данных
df = pd.read_csv('posture_features.csv')
X = df[['angle_ES', 'angle_SH', 'angle_HK', 'angle_KA']]  # признаки
y_true = df['label']  # истинные метки (1 = Good, 0 = Bad)

# 3. Предсказания
y_pred = model.predict(X)

# 4. Метрики
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n=== Оценка модели на всех данных ===")
print(f"Accuracy  : {acc:.2%}")
print(f"Precision : {prec:.2%}")
print(f"Recall    : {rec:.2%}")
print(f"F1-Score  : {f1:.2%}")
print("\nConfusion Matrix:")
print("            Predicted")
print("            Good  Bad")
print(f"Actual Good  {cm[1,1]:4d}  {cm[1,0]:4d}")
print(f"Actual Bad   {cm[0,1]:4d}  {cm[0,0]:4d}")

# 5. Подробный отчёт по каждому изображению
df['predicted'] = y_pred
df['correct'] = (df['predicted'] == df['label']).map({True: 'Yes', False: 'No'})
df['status_pred'] = df['predicted'].map({1: 'Good', 0: 'Bad'})
df['status_true'] = df['label'].map({1: 'Good', 0: 'Bad'})

print("\n=== Детальная таблица (первые 10 записей) ===")
print(df[['filename', 'status_true', 'status_pred', 'correct']].head(10).to_string(index=False))

# 6. Сохранение полной таблицы в Excel (если openpyxl установлен)
try:
    with pd.ExcelWriter('model_evaluation.xlsx') as writer:
        df[['filename', 'status_true', 'status_pred', 'correct'] +
           ['angle_ES', 'angle_SH', 'angle_HK', 'angle_KA']].to_excel(writer, sheet_name='Predictions', index=False)
        # Сохраняем метрики в отдельный лист
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [acc, prec, rec, f1]
        })
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    print("\nРезультаты сохранены в 'model_evaluation.xlsx'")
except ModuleNotFoundError:
    print("\nДля сохранения в Excel установите openpyxl: pip install openpyxl")
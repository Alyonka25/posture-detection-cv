import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Удаляем старый файл, если он есть (чтобы избежать повреждения)
if os.path.exists('summary_report.xlsx'):
    os.remove('summary_report.xlsx')
    print("Старый файл summary_report.xlsx удалён.")

# Загрузка модели и данных
model = joblib.load('posture_classifier.pkl')
df = pd.read_csv('posture_features.csv')

X = df[['angle_ES', 'angle_SH', 'angle_HK', 'angle_KA']]
y_true = df['label']
y_pred = model.predict(X)

# Метрики
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# ---- Таблица 1: Метрики ----
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [f'{acc:.2%}', f'{prec:.2%}', f'{rec:.2%}', f'{f1:.2%}']
}
df_metrics = pd.DataFrame(metrics_data)

# ---- Таблица 2: Матрица ошибок ----
confusion_data = {
    '': ['Actual Good', 'Actual Bad'],
    'Predicted Good': [tp, fp],
    'Predicted Bad': [fn, tn]
}
df_confusion = pd.DataFrame(confusion_data)

# ---- Сохранение в Excel ----
with pd.ExcelWriter('summary_report.xlsx', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
    df_confusion.to_excel(writer, sheet_name='Confusion Matrix', index=False)

print("Excel-файл 'summary_report.xlsx' успешно создан. Листы: Metrics, Confusion Matrix.")
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('posture_features.csv')

angle_cols = ['angle_ES', 'angle_SH', 'angle_HK', 'angle_KA']
df_dev = 180 - df[angle_cols].abs()   # отклонение от вертикали (0° = идеально)

df['max_dev'] = df_dev.max(axis=1)
df['primary_issue'] = df_dev.idxmax(axis=1).str.replace('angle_', '')

thresholds = [10, 15, 20, 25]
results = []

for th in thresholds:
    pred = (df_dev <= th).all(axis=1).astype(int)
    y_true = df['label'].values
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, pred, zero_division=0)
    results.append({
        'Threshold': th,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': f'{acc:.2%}',
        'Precision': f'{prec:.2%}',
        'Recall': f'{rec:.2%}',
        'Specificity': f'{spec:.2%}',
        'F1-Score': f'{f1:.2%}'
    })

metrics_df = pd.DataFrame(results)
print("\n=== Метрики для разных порогов ===")
print(metrics_df.to_string(index=False))

threshold_main = 20
pred_main = (df_dev <= threshold_main).all(axis=1).astype(int)
df['predicted_label'] = pred_main
df['status'] = df['predicted_label'].map({1: 'Good Posture', 0: 'Bad Posture'})

def get_misaligned(row, th=threshold_main):
    mis = []
    for col in angle_cols:
        dev = 180 - abs(row[col])
        if dev > th:
            mis.append(col.replace('angle_', ''))
    return ', '.join(mis) if mis else ''

df['misaligned'] = df.apply(lambda r: get_misaligned(r, threshold_main), axis=1)

detail_cols = ['filename', 'status'] + angle_cols + ['misaligned']
detail_df = df[detail_cols].copy()
detail_df.rename(columns={
    'angle_ES': 'ear-shoulder (deg)',
    'angle_SH': 'shoulder-hip (deg)',
    'angle_HK': 'hip-knee (deg)',
    'angle_KA': 'knee-ankle (deg)',
    'misaligned': 'Misaligned Segments'
}, inplace=True)

print("\n=== Детальная таблица (порог {}°) ===".format(threshold_main))
print(detail_df.to_string(index=False))

bad_df = df[df['predicted_label'] == 0]
issue_counts = bad_df['primary_issue'].value_counts()
print("\n=== Распределение первичных нарушений (Bad Posture) ===")
print(issue_counts)

def discrepancy_type(row, th=threshold_main):
    mis = get_misaligned(row, th)
    if mis == '':
        return 'No issue'
    parts = mis.split(', ')
    if len(parts) == 1:
        return 'Threshold Sensitivity'
    else:
        return 'Multiple issues'

df['discrepancy_type'] = df.apply(lambda r: discrepancy_type(r, threshold_main), axis=1)
print("\n=== Типы расхождений (порог {}°) ===".format(threshold_main))
print(df['discrepancy_type'].value_counts())

try:
    with pd.ExcelWriter('posture_report.xlsx') as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        detail_df.to_excel(writer, sheet_name='Details', index=False)
        issue_counts.to_excel(writer, sheet_name='Primary Issues')
        df['discrepancy_type'].value_counts().to_excel(writer, sheet_name='Discrepancy Types')
        df[['filename', 'label', 'predicted_label', 'status', 'primary_issue', 'misaligned', 'discrepancy_type'] + angle_cols].to_excel(writer, sheet_name='Full Data', index=False)
    print("\nОтчёт сохранён в 'posture_report.xlsx'")
except ModuleNotFoundError:
    print("\nДля сохранения в Excel установите openpyxl: pip install openpyxl")
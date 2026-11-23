import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('posture_features.csv')
X = df[['angle_ES', 'angle_SH', 'angle_HK', 'angle_KA']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Точность на тесте: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'posture_classifier.pkl')
print("Модель сохранена как posture_classifier.pkl")
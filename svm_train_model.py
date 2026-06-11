import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import joblib
import os

sklearn_verisi = load_breast_cancer()

df = pd.DataFrame(sklearn_verisi.data, columns=sklearn_verisi.feature_names)

df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

df['diagnosis'] = sklearn_verisi.target

print("Sınıf dağılımı (0 kötü 1 iyi huylu)")
print(df['diagnosis'].value_counts())
print("İlk 5 satır")
print(df.head())

ozellikler = ['mean radius', 'mean texture', 'mean perimeter',
              'mean area', 'mean smoothness']  # Veri setinden bu 5 özelliği kullandım

X = df[ozellikler]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(x_train_scaled, y_train)

tahmin = model.predict(x_test_scaled)
print(classification_report(y_test, tahmin,
      target_names=['Kotu Huylu', 'Iyi Huylu']))
dogruluk = accuracy_score(y_test, tahmin)
print(f"Doğruluk Oranı: %{round(dogruluk*100):.2f}")
os.makedirs('models', exist_ok=True)
joblib.dump(model,      'models/svm_model.joblib')
joblib.dump(scaler,     'models/scaler.joblib')
joblib.dump(ozellikler, 'models/ozellikler.joblib')
print("Kaydedildi.")

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

model = joblib.load('models/svm_model.joblib')
scaler = joblib.load('models/scaler.joblib')
ozellikler = joblib.load('models/ozellikler.joblib')

sklearn_verisi = load_breast_cancer()

df = pd.DataFrame(sklearn_verisi.data, columns=sklearn_verisi.feature_names)

df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

df['diagnosis'] = sklearn_verisi.target

iyi_satir = df[df['diagnosis'] == 1].iloc[0]   # Iyi huylu ornek
kotu_satir = df[df['diagnosis'] == 0].iloc[0]   # Kotu huylu ornek


def tahmin_et(radius, texture, perimeter, area, smoothness):
    yeni_degerler = np.array([[radius, texture, perimeter, area, smoothness]])
    veri_scaled = scaler.transform(yeni_degerler)
    tahmin = model.predict(veri_scaled)[0]
    olasilik = model.predict_proba(veri_scaled)[0]
    if tahmin == 1:
        sonuc = "✅ Iyi Huylu"
    else:
        sonuc = "⚠️ Kotu Huylu"

    iyi_olasilik = round(olasilik[1] * 100, 1)
    kotu_olasilik = round(olasilik[0] * 100, 1)

    return f"Sonuc: {sonuc}\n\n Iyi Huylu: %{iyi_olasilik}\n\nKotu Huylu: %{kotu_olasilik}"


arayuz = gr.Interface(
    fn=tahmin_et,
    inputs=[
        gr.Number(label="Radius Mean",      value=round(
            iyi_satir['mean radius'], 2)),
        gr.Number(label="Texture Mean",     value=round(
            iyi_satir['mean texture'], 2)),
        gr.Number(label="Perimeter Mean",   value=round(
            iyi_satir['mean perimeter'], 2)),
        gr.Number(label="Area Mean",        value=round(
            iyi_satir['mean area'], 2)),
        gr.Number(label="Smoothness Mean",  value=round(
            iyi_satir['mean smoothness'], 4)),
    ],
    outputs=gr.Markdown(label="Sonuç"),
    title="Kanser Risk Tahmini",
    description="Tümör ölçümlerini girin, model iyi huylu mu kötü huylu mu tahmin etsin.",
    examples=[
        [round(iyi_satir['mean radius'], 2),  round(iyi_satir['mean texture'], 2),
         round(iyi_satir['mean perimeter'], 2), round(
             iyi_satir['mean area'], 2),
         round(iyi_satir['mean smoothness'], 4)],

        [round(kotu_satir['mean radius'], 2), round(kotu_satir['mean texture'], 2),
         round(kotu_satir['mean perimeter'], 2), round(
             kotu_satir['mean area'], 2),
         round(kotu_satir['mean smoothness'], 4)],
    ],
    theme=gr.themes.Soft()
)

arayuz.launch()

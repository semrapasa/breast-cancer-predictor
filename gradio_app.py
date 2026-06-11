import gradio as gr
import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')
ozellikler = joblib.load('models/ozellikler.pkl')

# Örnek değerleri de URL'den çek
df = pd.read_csv(
    "https://raw.githubusercontent.com/dphi-official/Datasets/master/breast_cancer.csv")
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})

iyi_satir = df[df['diagnosis'] == 1].iloc[0]   # iyi huylu örnek
kotu_satir = df[df['diagnosis'] == 0].iloc[0]   # kötü huylu örnek


def tahmin_et(radius, texture, perimeter, area, smoothness):
    yeni_degerler = np.array([[radius, texture, perimeter, area, smoothness]])
    veri_scaled = scaler.transform(yeni_degerler)
    tahmin = model.predict(veri_scaled)[0]
    olasilik = model.predict_proba(veri_scaled)[0]
    if tahmin == 1:
        sonuc = "✅ İyi Huylu"
    else:
        sonuc = "⚠️ Kötü Huylu"

    iyi_olasilik = round(olasilik[1] * 100, 1)
    kotu_olasilik = round(olasilik[0] * 100, 1)

    return f"**Sonuç:** {sonuc}\n\n**İyi Huylu:** %{iyi_olasilik}\n\n**Kötü Huylu:** %{kotu_olasilik}"


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
        [round(iyi_satir['radius_mean'], 2),  round(iyi_satir['texture_mean'], 2),
         round(iyi_satir['perimeter_mean'], 2), round(
             iyi_satir['area_mean'], 2),
         round(iyi_satir['smoothness_mean'], 4)],

        [round(kotu_satir['radius_mean'], 2), round(kotu_satir['texture_mean'], 2),
         round(kotu_satir['perimeter_mean'], 2), round(
             kotu_satir['area_mean'], 2),
         round(kotu_satir['smoothness_mean'], 4)],
    ],
    theme=gr.themes.Soft()
)

arayuz.launch()

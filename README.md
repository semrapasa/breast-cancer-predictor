<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&amp;color=gradient&amp;customColorList=120,40,140,50,1,160&amp;height=220&amp;section=header&amp;text=breast-cancer-predictor&amp;fontSize=42&amp;fontColor=fff&amp;animation=twinkling&amp;fontAlignY=40&amp;desc=Gradio%20%7C%20Scikit-Learn%20%7C%20Support%20Vector%20Machines%20(SVM)%20%7C%20Oncology%20Analytics&amp;descAlignY=62&amp;descSize=15" />
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/semrapasa/breast-cancer-predictor" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Canlı%20Demo-Hugging%20Face%20Spaces-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Gradio-FF5A00?style=for-the-badge&amp;logo=gradio&amp;logoColor=white" />
  <img src="https://img.shields.io/badge/Algorithm-SVM%20(Linear/RBF)-0052CC?style=for-the-badge&amp;logo=scikit-learn&amp;logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-Breast%20Cancer%20Wisconsin-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/UI__Theme-Gradio__Soft-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Model__Format-Joblib-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

## 🔬 Proje Özeti

**breast-cancer-predictor**, klinik göğüs kanseri veri kümesi (*Wisconsin Breast Cancer Dataset*) parametreleri üzerine eğitilmiş bir **Destek Vektör Makineleri (SVM)** modelini joblib ile dumplar. Uygulama çalıştığında halihazırda eğitilmiş model ve scaler çağırılır. Hugging Face Gradio ile deploy edilen ve interface kazanan bu uygulama girilen veriler ile teşhis oranı oluşturan bir yapay zeka özelliği taşır.

Proje, karmaşık matematiksel veri ölçeklendirme (StandardScaler) ve çok boyutlu sınıflandırma tahminlerini, son kullanıcının veya bir onkoloğun saniyeler içinde kullanabileceği interaktif bir **Gradio Soft** web arayüzüne dönüştürerek kolaylık sağlar.

---

## 🎨 Canlı Arayüz Önizlemesi & Çalışma Örnekleri

Sistem, Gradio altyapısı sayesinde dinamik olasılık dağılımlarını anlık olarak render eder. Aşağıda sistemin iyi huylu ve kötü huylu tümör metriklerine göre gerçekleştirdiği canlı tahminlerin akış şeması ve arayüz yapısı yer almaktadır:


### 📊 Sistem Çıktı Örnekleri

| Giriş Örneği Tipi | Radius / Texture / Area Ölçümleri | Model Tahmin Çıktısı (Markdown Render) |
| :--- | :--- | :--- |
| 🟢 **Örnek 1 (İyi Huylu)** | `13.54 / 14.36 / 566.3 / 0.09779` | **Sonuc:** ✅ Iyi Huylu <br> İyi Huylu: %94.2 <br> Kötü Huylu: %5.8 |
| 🔴 **Örnek 2 (Kötü Huylu)** | `17.99 / 10.38 / 1001.0 / 0.1184` | **Sonuc:** ⚠️ Kotu Huylu <br> İyi Huylu: %1.4 <br> Kötü Huylu: %98.6 |

---

## ⚙️ Teknik Altyapı ve Karar Mekanizması

Uygulama, ham verileri işleyip sınıflandırma olasılıklarına dönüştürürken doğrusal ve radyal tabanlı bir karar sınırı (Decision Boundary) mekanizması kullanır. Bu verinin bir üst boyuta taşınarak net ayrılmasını sağlar.


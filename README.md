<div align="center">

# 🩺 Diabetes AI Predictor

### An intelligent, interactive web application for diabetes risk prediction  
### powered by **Gaussian Naïve Bayes** and the **Pima Indians Diabetes Dataset**

<br>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br>

<!-- Replace the URL below with your actual deployed app link -->
**[🚀 Live Demo](https://your-app.streamlit.app)** &nbsp;·&nbsp;
**[📊 Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)** &nbsp;·&nbsp;
**[🐛 Report Bug](../../issues)** &nbsp;·&nbsp;
**[✨ Request Feature](../../issues)**

</div>

---

## 📸 Screenshots

> *A dark-themed, interactive dashboard with real-time predictions, probability gauges, radar charts, and Gaussian distribution visualizations.*

| 🔮 Prediction Tab | 📊 Feature Analysis |
|:-:|:-:|
| ![Prediction](https://placehold.co/600x350/0f0c29/818cf8?text=Prediction+%26+Gauge) | ![Feature Analysis](https://placehold.co/600x350/0f0c29/4ade80?text=Feature+Analysis) |

| 🧠 Model Insights | 🎛️ Sidebar Controls |
|:-:|:-:|
| ![Model Insights](https://placehold.co/600x350/0f0c29/f87171?text=Model+Insights) | ![Sidebar](https://placehold.co/600x350/1e1b4b/c4b5fd?text=Patient+Parameters) |

> 💡 **Tip:** Replace placeholder images above with real screenshots of your running app.  
> Take a screenshot → upload to your repo under `assets/` → update the paths like `![Prediction](assets/prediction.png)`

---

## ✨ Features

- 🔮 **Real-time Prediction** — Instant diabetes risk classification as you move the sliders  
- 📊 **Probability Gauge** — Animated semicircular gauge showing diabetes likelihood (0–100%)  
- 🎯 **Risk Badges** — Dynamic LOW / MODERATE / HIGH RISK labels based on probability thresholds  
- 🕸️ **Radar Chart** — Normalized comparison of your values vs. diabetic and non-diabetic population means  
- 📉 **Deviation Analysis** — Bar chart showing how far each feature deviates from the diabetic mean  
- 🔔 **Gaussian Distributions** — Visual probability density curves per feature for both classes  
- 🧬 **Feature Means Comparison** — Horizontal grouped bar chart of class-wise feature averages  
- 🍩 **Class Distribution Donut** — Training data split visualization (267 Non-DM vs 347 DM)  
- 🌑 **Dark Gradient UI** — Custom CSS with deep purple/indigo theme built for Streamlit  
- 📋 **Patient Summary Table** — Per-feature proximity indicator to diabetic vs. non-diabetic averages  

---

## 🏗️ Project Structure

```
diabetes-ai-predictor/
│
├── app.py                  # Main Streamlit application
├── Diabates.joblib         # Pre-trained Gaussian Naïve Bayes model
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── assets/                 # Screenshots for README (add your own)
    ├── prediction.png
    ├── feature_analysis.png
    └── model_insights.png
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Algorithm** | Gaussian Naïve Bayes |
| **Library** | scikit-learn |
| **Dataset** | Pima Indians Diabetes (UCI) |
| **Training Samples** | 614 |
| **Diabetic Cases** | 347 (56.5%) |
| **Non-Diabetic Cases** | 267 (43.5%) |
| **Input Features** | 8 clinical biomarkers |
| **Output** | Binary (0 = Non-Diabetic, 1 = Diabetic) |

### 📥 Input Features

| # | Feature | Description | Range |
|---|---|---|---|
| 1 | **Pregnancies** | Number of times pregnant | 0 – 17 |
| 2 | **Glucose** | Plasma glucose concentration (mg/dL) | 0 – 200 |
| 3 | **Blood Pressure** | Diastolic blood pressure (mm Hg) | 0 – 122 |
| 4 | **Skin Thickness** | Triceps skin fold thickness (mm) | 0 – 99 |
| 5 | **Insulin** | 2-hour serum insulin (μU/mL) | 0 – 846 |
| 6 | **BMI** | Body Mass Index (kg/m²) | 0 – 67.1 |
| 7 | **Diabetes Pedigree Function** | Hereditary risk score | 0.078 – 2.42 |
| 8 | **Age** | Patient age in years | 21 – 81 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/diabetes-ai-predictor.git
cd diabetes-ai-predictor
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
streamlit run app.py
```

**5. Open your browser**

```
http://localhost:8501
```

---

## 📦 Requirements

```txt
streamlit>=1.28.0
scikit-learn>=1.0.0
joblib>=1.2.0
numpy>=1.23.0
plotly>=5.15.0
```

Save the above as `requirements.txt` in your project root.

---

## 🖥️ App Walkthrough

### 🎛️ Sidebar — Patient Input
Adjust 8 clinical sliders to set patient values. The prediction updates automatically on every change.

### Tab 1 — 🔮 Prediction
- **Result Card** flips between green (Non-Diabetic) and red (Diabetic)
- **Gauge Meter** shows diabetes probability from 0–100%
- **Probability Bar Chart** displays both class probabilities side by side
- **Patient Summary** shows each feature value alongside diabetic/non-diabetic averages with a proximity indicator

### Tab 2 — 📊 Feature Analysis
- **Radar Chart** overlays your values against both class means on a normalized 0–1 scale
- **Grouped Bar + Line Chart** compares your raw values vs. both class averages
- **Deviation Chart** shows how much each feature deviates above or below the diabetic mean

### Tab 3 — 🧠 Model Insights
- **Donut Chart** of the training data class split
- **Horizontal Grouped Bar** of feature means for each class
- **Gaussian Distribution Subplots** — 8 density curve pairs (one per feature) showing the model's learned probability distributions

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push your project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repository
4. Set **Main file path** to `app.py`
5. Click **Deploy** — your live URL will be ready in ~2 minutes

> Update the **Live Demo** badge link at the top of this README with your deployed URL.

---

## ⚕️ Medical Disclaimer

> This application is built for **educational and demonstration purposes only**.  
> It does **not** constitute medical advice, diagnosis, or treatment.  
> Always consult a qualified healthcare professional for any health-related decisions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes) — Pima Indians Diabetes Dataset
- [Streamlit](https://streamlit.io) — For making beautiful ML apps effortless
- [Plotly](https://plotly.com) — For stunning interactive charts
- [scikit-learn](https://scikit-learn.org) — For the Gaussian Naïve Bayes implementation

---

<div align="center">

Made with ❤️ using Python & Streamlit

⭐ **Star this repo if you found it helpful!** ⭐

</div>

# 🌾 AgroSage — Smart Crop Recommendation System

> AI-powered crop advisory platform combining soil analysis, plant disease detection, and real-time weather intelligence.

---

## Features

| Module | Description |
|---|---|
| **🌱 Crop Recommender** | Predicts the best crop based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall using an ensemble ML model. |
| **🍃 Disease Detector** | Identifies plant diseases from leaf images using a CNN classifier built with TensorFlow/Keras. |
| **🌦️ Weather Analyst** | Fetches real-time weather data via OpenWeatherMap API and displays climate trends with interactive Plotly charts. |
| **📊 Dashboard** | Central overview with key stats, recent activity, and system health indicators. |

---

## Project Structure

```
AgroSage/
├── app.py                         # Streamlit entry point
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template
├── README.md
│
├── config/
│   └── settings.py                # Centralised constants & config
│
├── assets/
│   ├── logo.png                   # App logo
│   └── styles.css                 # Custom Streamlit CSS overrides
│
├── data/
│   ├── crop_recommendation.csv    # Training dataset for crop model
│   └── disease_classes.json       # Label → class-name mapping
│
├── models/
│   ├── crop_recommender.pkl       # Serialised sklearn pipeline
│   └── disease_detector.h5        # Trained Keras CNN weights
│
├── pages/
│   ├── 1_🌱_Crop_Recommender.py
│   ├── 2_🍃_Disease_Detector.py
│   └── 3_🌦️_Weather_Analyst.py
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # CSV / JSON loading helpers
│   ├── model_utils.py             # Model load / predict wrappers
│   ├── weather_api.py             # OpenWeatherMap client
│   ├── image_processing.py        # Leaf image pre-processing
│   └── ui_components.py           # Reusable Streamlit widgets
│
└── training/
    ├── train_crop_model.py        # Crop model training script
    └── train_disease_model.py     # Disease CNN training script
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/AgroSage.git
cd AgroSage

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# → Add your OpenWeatherMap API key

# 4. Train models (first time only)
python training/train_crop_model.py
python training/train_disease_model.py

# 5. Launch
streamlit run app.py
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `OWM_API_KEY` | OpenWeatherMap API key for weather data |

---

## Tech Stack

- **UI**: Streamlit 1.41
- **ML**: scikit-learn 1.6, TensorFlow 2.18
- **Viz**: Plotly 5.24
- **Image**: Pillow 11.1
- **Data**: pandas 2.2, NumPy 1.26

---

## Build Phases

| Phase | Scope |
|---|---|
| **1** | Architecture, folder structure, dependencies *(this phase)* |
| **2** | Config, assets, CSS theming, data files |
| **3** | Utility modules (`utils/`) |
| **4** | Model training scripts (`training/`) |
| **5** | Dashboard home page (`app.py`) |
| **6** | Crop Recommender page |
| **7** | Disease Detector page |
| **8** | Weather Analyst page |
| **9** | Integration testing & polish |

---

## License

MIT

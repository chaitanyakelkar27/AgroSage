# AgroSage — Smart Crop Advisory Platform

> AI-powered crop recommendation, plant disease detection, and weather intelligence — all in one Streamlit dashboard.

---

## Features

| Module | Description |
|---|---|
| **Crop Recommender** | Predicts the best crop from soil nutrient levels (N, P, K), temperature, humidity, pH, and rainfall using a Random Forest ensemble tuned with GridSearchCV. |
| **Disease Detector** | Identifies plant diseases from leaf images using a TensorFlow/Keras CNN classifier with confidence scores. |
| **Weather Analyst** | Real-time weather data via OpenWeatherMap API with interactive Plotly climate charts. |
| **Dashboard** | System overview with live status checks, dataset metrics, and a getting-started guide. |

---

## Quick Start

### 1 · Clone the repository

```bash
git clone https://github.com/your-org/AgroSage.git
cd AgroSage
```

### 2 · Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.41.1 | Web UI framework |
| scikit-learn | 1.6.1 | Crop recommendation model |
| tensorflow | 2.18.0 | Disease detection CNN |
| pandas | 2.2.3 | Data processing |
| numpy | 1.26.4 | Numerical computation |
| plotly | 5.24.1 | Interactive charts |
| joblib | 1.4.2 | Model serialisation |
| Pillow | 11.1.0 | Image processing |
| requests | 2.32.3 | HTTP / API calls |
| python-dotenv | 1.0.1 | Environment variables |

### 4 · Place datasets

Put the following files inside the `data/` folder:

| File | Rows | Description |
|---|---|---|
| `Crop_recommendation.csv` | 2,200 | 7 soil/weather features + 22 crop labels (100 samples each) |
| `DailyDelhiClimate.csv` | ~1,500 | Historical daily temperature, humidity, wind speed, pressure |

The crop CSV must have these exact columns:
```
N, P, K, temperature, humidity, ph, rainfall, label
```

### 5 · Train the crop model

```bash
python training/train_crop_model.py
```

This runs a full ML pipeline:
- Loads and cleans the crop dataset
- 80/20 stratified train/test split
- StandardScaler feature normalisation
- GridSearchCV over a Random Forest classifier (36 hyperparameter combos × 5 folds)
- KMeans clustering for similar-crop suggestions
- Saves 5 artifacts to `models/`:
  - `crop_model.pkl` — best Random Forest estimator (~10 MB)
  - `crop_scaler.pkl` — fitted StandardScaler
  - `crop_classes.pkl` — ordered list of 22 crop names
  - `crop_kmeans.pkl` — fitted KMeans model
  - `crop_cluster_map.pkl` — cluster → crop mapping

**Training time:** ~30–60 seconds on a modern CPU.

### 6 · Configure environment (optional)

```bash
cp .env.example .env
```

Edit `.env` and add your OpenWeatherMap API key. This is only required for the Weather Analyst page:

```
OWM_API_KEY=your_actual_api_key
```

Get a free key at [openweathermap.org/api](https://openweathermap.org/api).

### 7 · Train the disease model

```bash
python training/train_disease_model.py
```

This pipeline:
- Loads image classes from `data/plant_disease_dataset/train` and `data/plant_disease_dataset/valid`
- Trains or resumes a denoising autoencoder (`autoencoder.h5`)
- Trains a CNN disease classifier (`disease_cnn.h5`)
- Saves class mappings (`disease_classes.pkl`)

Recent completed run summary:
- Best validation accuracy: **97.90%**
- Best epoch restored: **epoch 7**

### 8 · Evaluate disease model (recommended)

```bash
python training/evaluate_disease_model.py --samples 300
```

Optional flags:
- `--samples` number of validation images to evaluate
- `--seed` reproducible random sampling seed

### 9 · Launch the app

```bash
streamlit run Home.py
```

The app opens in your default browser at `http://localhost:8501`. Use the sidebar to navigate between modules.

---

## Project Structure

```
AgroSage/
├── Home.py                            # Streamlit entry point & dashboard
├── training/
│   ├── train_crop_model.py            # Crop model training pipeline
│   ├── train_disease_model.py         # Disease model training pipeline
│   └── evaluate_disease_model.py      # Disease evaluation script
├── requirements.txt                   # Pinned dependencies
├── .env.example                       # Environment variable template
├── README.md
│
├── config/
│   └── settings.py                    # Centralised constants & config
│
├── assets/
│   └── style.css                      # Global CSS design system
│
├── data/
│   ├── Crop_recommendation.csv        # Crop training dataset
│   └── DailyDelhiClimate.csv          # Historical climate data
│
├── models/
│   ├── crop_model.pkl                 # Trained Random Forest
│   ├── crop_scaler.pkl                # Fitted StandardScaler
│   ├── crop_classes.pkl               # Crop label list
│   ├── crop_kmeans.pkl                # KMeans clustering model
│   ├── crop_cluster_map.pkl           # Cluster → crop mapping
│   ├── autoencoder.h5                 # Leaf image denoising model
│   ├── disease_cnn.h5                 # Disease classifier model
│   └── disease_classes.pkl            # Disease class label mapping
│
├── pages/
│   ├── 1_Crop_Recommender.py
│   ├── 2_Disease_Detector.py
│   └── 3_Weather_Analyst.py
│
└── utils/
    ├── __init__.py                    # Package exports & CSS loader
    ├── crop_predictor.py              # Model loading & prediction
  ├── disease_predictor.py           # Disease model loading & inference
  └── weather_utils.py               # Climate CSV & weather helpers
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **UI** | Streamlit 1.41 |
| **ML** | scikit-learn 1.6 · Random Forest · GridSearchCV · KMeans |
| **DL** | TensorFlow 2.18 · Keras |
| **Visualisation** | Plotly 5.24 |
| **Data** | pandas 2.2 · NumPy 1.26 |
| **Serialisation** | joblib 1.4 |

---

## GPU Note

- **Crop Recommender** uses scikit-learn (CPU only) — no GPU needed.
- **Disease Detector** uses TensorFlow, which will automatically use a CUDA-compatible GPU if available. Training the CNN benefits from a GPU; inference works fine on CPU (~1–2s per image).
- If you don't have a GPU, install the CPU-only TensorFlow variant:
  ```bash
  pip install tensorflow-cpu==2.18.0
  ```

---

## License

MIT

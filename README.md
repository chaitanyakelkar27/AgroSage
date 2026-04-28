# AgroSage

AgroSage is a Streamlit-based smart crop advisory platform that brings together crop recommendation, plant disease detection, and weather analysis in a single dashboard. It is designed to help farmers and agronomy teams make faster, data-driven decisions using a mix of classical machine learning, deep learning, and climate analytics.

## Features

- Crop recommendation based on soil nutrients and weather inputs such as N, P, K, temperature, humidity, pH, and rainfall.
- Plant disease detection from leaf images using a TensorFlow/Keras CNN with confidence scoring.
- Weather analysis with historical climate trends and OpenWeatherMap-backed forecasts.
- A unified multi-page Streamlit dashboard with a reusable sidebar, shared styling, and modular page design.
- Training and evaluation scripts for regenerating the crop and disease models.

## Tech Stack

| Area | Stack |
|---|---|
| UI | Streamlit |
| Machine Learning | scikit-learn, Random Forest, GridSearchCV, KMeans |
| Deep Learning | TensorFlow, Keras |
| Data Processing | pandas, NumPy |
| Visualization | Plotly |
| Imaging | Pillow |
| Model Persistence | joblib, HDF5 artifacts |
| API Access | requests |

## Concepts Used

| Concept | Where It Appears |
|---|---|
| Supervised classification | Crop recommendation and disease detection |
| Feature scaling | Crop model preprocessing with StandardScaler |
| Hyperparameter tuning | GridSearchCV for the crop model |
| Clustering | Similar crop suggestions using KMeans |
| Image preprocessing | Leaf upload, cleaning, and classifier input preparation |
| Confidence analysis | Ranked disease prediction outputs |
| Time-series analysis | Historical weather and climate charts |
| Caching | Streamlit resource caching for model loading |
| Multi-page app design | Shared navigation across dashboard pages |

## Libraries

| Library | Purpose |
|---|---|
| streamlit | Web app framework and UI rendering |
| scikit-learn | Crop model training and inference |
| tensorflow | Disease classifier and autoencoder |
| pandas | CSV loading and tabular manipulation |
| numpy | Numerical operations |
| plotly | Interactive charts and visualizations |
| Pillow | Image handling for uploads |
| requests | Weather API requests |
| joblib | Saved model and scaler loading |
| python-dotenv | Environment variable support |
| streamlit-option-menu | Navigation/UI helper |
| streamlit-lottie | Animation support |

## Getting Started

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and add your OpenWeatherMap API key:

```bash
copy .env.example .env
```

The weather page expects:

```env
OWM_API_KEY=your_api_key_here
```

### 4. Verify data files

Place the required datasets under `data/`:

- `Crop_recommendation.csv`
- `DailyDelhiClimate.csv`
- `plant_disease_dataset/` or `Plant Village Dataset/` with training and validation images

### 5. Train or refresh the models

```bash
python training/train_crop_model.py
python training/train_disease_model.py
```

Optional evaluation:

```bash
python training/evaluate_disease_model.py --samples 300
```

### 6. Launch the app

```bash
streamlit run Home.py
```

The app opens in your browser at `http://localhost:8501`.

## Project Structure

```text
AgroSage/
├── Home.py
├── pages/
│   ├── 1_Crop_Recommender.py
│   ├── 2_Disease_Detector.py
│   └── 3_Weather_Analyst.py
├── utils/
│   ├── __init__.py
│   ├── crop_predictor.py
│   ├── disease_predictor.py
│   └── weather_utils.py
├── training/
│   ├── train_crop_model.py
│   ├── train_disease_model.py
│   ├── evaluate_disease_model.py
│   └── generate_disease_report_figures.py
├── config/
│   └── settings.py
├── assets/
│   └── style.css
├── data/
├── models/
├── artifacts/
│   └── report/
├── requirements.txt
└── README.md
```

## Model Artifacts

The repository includes trained artifacts under `models/`:

- `crop_model.pkl`
- `crop_scaler.pkl`
- `crop_classes.pkl`
- `crop_kmeans.pkl`
- `crop_cluster_map.pkl`
- `autoencoder.h5`
- `disease_cnn.h5`
- `disease_classes.pkl`

## Notes

- Crop recommendation runs on CPU and does not require a GPU.
- Disease model training benefits from a CUDA-capable GPU, but inference works on CPU.
- If you want a CPU-only TensorFlow installation, use `tensorflow-cpu` instead of `tensorflow`.

## License

MIT

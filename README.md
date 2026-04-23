# Brain Tumor Prediction System

A professional, production-ready full-stack machine learning application that predicts the likelihood of brain tumors being benign or malignant based on clinical characteristics. Built using FastAPI for an ultra-fast backend, XGBoost as the core prediction engine, and Jinja2 for dynamic frontend rendering.

## Features

- **XGBoost Machine Learning Model**: Uses advanced gradient boosting properly protected against data leakage using a Scikit-Learn Pipeline.
- **7 Clinical Inputs**: Evaluates complex medical properties (Age, Tumor Size, Survival Rate, Tumor Growth Rate, Stage, Tumor Density, Edema Size).
- **Confidence Scoring**: Dynamically calculates prediction probability boundaries displaying confidence visually per result.
- **Feature Importance Tracking**: Integrates XGboost importance plotting globally mapping insights transparently to the user interface natively.
- **Responsive Web UI**: Fully responsive, sleek dark mode frontend powered without any invasive frameworks directly leveraging Vanilla CSS via Jinja.

## Installation & Setup

1. **Clone the repository** and ensure you have a Python environment running (3.9+ recommended).
   
2. **Install all dependencies natively**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the local ML Model**:
   Execute the processing script to handle parsing the local dataset and generating the cached model environment schemas.
   ```bash
   python train_model.py
   ```
   
4. **Launch the Web Application Backend**:
   Run the ASGI-compatible server configurations natively.
   ```bash
   python main.py
   ```
   *Navigate to `http://localhost:8000` to interact with the application.*

## Deployment

This platform configuration is production-ready via `uvicorn`. The existing configurations contain dynamic port bindings compatible automatically via a Heroku deployment scaling with an assigned `Procfile`.

> **Disclaimer**: This tool and associated machine learning architectures represent mock models designed for educational purposes only. Do NOT utilize this model for actual medical diagnosis.

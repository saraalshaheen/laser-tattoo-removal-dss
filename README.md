# Clinical Laser Tattoo Removal DSS

A web-based decision-support prototype for recommending laser tattoo removal parameters using trained machine-learning models.

## Scope
This application is part of the thesis project **Developing a Smart Guideline System for Laser Therapy**. It is a guideline-support prototype and **not** an autonomous medical decision tool.

## Inputs
- Skin type
- Tattoo color
- Tattoo size
- Tattoo age in years and months
- Tattoo type
- Laser type
- Repetition rate (Hz)

## Outputs
- Wavelength
- Energy
- Pulse duration
- Total pulses

## Main files
- `app.py`: Streamlit application
- `models/laser_dss_model_bundle.joblib`: trained model bundle
- `requirements.txt`: Python dependencies
- `data/laser_dss_records.db`: created automatically for patient/session records

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Upload the repository to GitHub and deploy it using Streamlit Community Cloud. The application does not retrain the models during deployment; it loads the saved model bundle.

## Clinical note
All outputs must be reviewed by a dermatologist or laser specialist before any clinical use.

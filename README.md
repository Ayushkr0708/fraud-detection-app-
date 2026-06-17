# Financial Fraud Detection App

A Streamlit dashboard for financial fraud detection using transaction graph analysis, anomaly detection, fraud risk scoring, and model-performance visualizations.

## Files Required for App Upload

Upload these three CSV files inside the app:

1. `elliptic_txs_features.csv` or `elliptic_txs_features_under200mb.csv`
2. `elliptic_txs_edgelist.csv`
3. `elliptic_txs_classes.csv`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Upload this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Select your GitHub repository.
4. Set main file path as `app.py`.
5. Click Deploy.

## Note

This deployment version removes Colab commands, ngrok code, and training-only code. It is ready for Streamlit Community Cloud.

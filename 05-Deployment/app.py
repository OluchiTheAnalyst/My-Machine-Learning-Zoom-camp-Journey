from fastapi import FastAPI
import pickle

app = FastAPI()

# Load the model once when the app starts
with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

@app.get("/")
def home():
    return {"message": "FastAPI is working fine!"}

@app.post("/predict")
def predict(client: dict):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    return {"probability": float(y_pred)}

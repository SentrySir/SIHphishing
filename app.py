from fastapi import FastAPI
from pydantic import BaseModel
from url import extract_features
import pandas as pd
import tensorflow as tf

app = FastAPI()

# Load your machine learning model
model_main = tf.keras.models.load_model('my_model.h5')

class URLInput(BaseModel):
    url: str

class PredictionOutput(BaseModel):
    prediction: float

@app.post("/predict/")
async def predict_url(url_input: URLInput):
    try:
        # Preprocess the URL and extract features
        test_url = url_input.url
        if not test_url.startswith("http://") and not test_url.startswith("https://"):
            test_url = "http://" + test_url

        features = extract_features(test_url)
        df = pd.DataFrame([features])

        # Make a prediction using the loaded model
        prediction = model_main.predict(df)
        prediction = prediction[0][0]

        return {"prediction": prediction}
    except Exception as e:
        return {"prediction": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

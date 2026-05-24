from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from standardizer import Standardizer
from classify import classify

app = FastAPI(title="Fast AI Server")

class Input(BaseModel):
    daily_travel_time: float
    vehicle_ownership: str
    location_type: str
    nearby_industries: float
    green_space_access: str
    home_air_quality: float
    work_location_type: str
    smoker_in_household: bool
    noise_pollution_level: float
    use_of_air_purifiers: bool
    awareness_level: str
    years_in_location: float

@app.post("/predict")
def predict(input: Input):
    data_dict = input.model_dump()
    data = pd.DataFrame([data_dict])
    res = Standardizer(data)
    prediction = classify(res)
    print(prediction)
    return {"prediction": prediction}

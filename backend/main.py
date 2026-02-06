# from fastapi import FastAPI
# import pickle
# import numpy as np

# app = FastAPI()

# model = pickle.load(open("../model/model.pkl", "rb"))
# scaler = pickle.load(open("../model/scaler.pkl", "rb"))

# @app.get("/")
# def home():
#     return {"message": "Employee Attrition API running"}

# @app.post("/predict")
# def predict(data: dict):
#     features = np.array([[ 
#         data["Age"],
#         data["MonthlyIncome"],
#         data["JobLevel"],
#         data["TotalWorkingYears"]
#     ]])

#     scaled = scaler.transform(features)
#     result = model.predict(scaled)

#     return {"Attrition": "Yes" if result[0] == 1 else "No"}
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("../model/model.pkl", "rb"))
scaler = pickle.load(open("../model/scaler.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Employee Attrition API Running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[ 
        data["Age"],
        data["DailyRate"],
        data["DistanceFromHome"],
        data["Education"],
        data["EnvironmentSatisfaction"],
        data["JobInvolvement"],
        data["JobLevel"],
        data["JobSatisfaction"],
        data["MonthlyIncome"],
        data["NumCompaniesWorked"],
        data["PercentSalaryHike"],
        data["PerformanceRating"],
        data["TotalWorkingYears"],
        data["WorkLifeBalance"],
        data["YearsAtCompany"]
    ]])

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    return {"Attrition": "Yes" if prediction[0] == 1 else "No"}
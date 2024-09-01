# main.py
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import pickle
import datetime
import traceback
from nltk.corpus import stopwords

# Ensure you have downloaded the stopwords dataset
# nltk.download('stopwords')

# List of stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from a sentence
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def date_formater(post_date):
    try:
        datetime_obj = datetime.datetime.strptime(post_date, "%Y-%m-%d %H:%M:%S")
        post_year = datetime_obj.year
        post_month = datetime_obj.month
        post_day = datetime_obj.day
        return {"year": post_year, "month": post_month, "day": post_day}
    except ValueError:
        raise ValueError("Invalid date format. Please provide a string in YYYY-MM-DD HH:MM:SS format.")

# Define the Pydantic model
class AdRequest(BaseModel):
    post_date: str
    country: str
    call_to_action: str
    likes: int
    comments: int
    ad_position: str
    cleaned_ad_title: str
    cleaned_newsfeed_description: str
    cleaned_ad_text: str
    duration_days: int

def preprocess(ad_request: AdRequest):
    post_date_info = date_formater(ad_request.post_date)
    post_day = post_date_info.get("day")
    post_year = post_date_info.get("year")
    post_month = post_date_info.get("month")
    country = ad_request.country if ad_request.country else "United States"
    call_to_action = ad_request.call_to_action if ad_request.call_to_action else "Try free"
    likes = ad_request.likes if ad_request.likes else 0
    comments = ad_request.comments if ad_request.comments else 0
    ad_position = ad_request.ad_position if ad_request.ad_position else "SIDE"
    cleaned_ad_title = remove_stopwords(ad_request.cleaned_ad_title) if ad_request.cleaned_ad_title else "10 Week Program"
    cleaned_newsfeed_description = remove_stopwords(ad_request.cleaned_newsfeed_description) if ad_request.cleaned_newsfeed_description else "Hello"
    cleaned_ad_text = remove_stopwords(ad_request.cleaned_ad_text) if ad_request.cleaned_ad_text else "Run your socially"
    duration_days = ad_request.duration_days if ad_request.duration_days else 0
    
    return {
        "post_day": post_day,
        "post_year": post_year,
        "post_month": post_month,
        "country": country,
        "call_to_action": call_to_action,
        "likes": likes,
        "comments": comments,
        "ad_position": ad_position,
        "cleaned_ad_title": cleaned_ad_title,
        "cleaned_newsfeed_description": cleaned_newsfeed_description,
        "cleaned_ad_text": cleaned_ad_text,
        "duration_days": duration_days
    }

def prediction(payload):
    # Convert the single payload to a pandas DataFrame
    single_payload_df = pd.DataFrame([payload])

    # Use the loaded model to make predictions
    single_prediction_scaled_log = Popularity_model.predict(single_payload_df)

    print(f'Predicted Views (scaled log): {single_prediction_scaled_log[0]}')
    return single_prediction_scaled_log[0]

# Load Model
with open(r"C:\Users\glb\Downloads\Faisal Data\Popularity_model2.0.pkl", 'rb') as f:
    Popularity_model = pickle.load(f)

# Create the FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "This API gives the Popularity_Index from the machine learning model", "route": "/popularity/?"}

@app.post("/popularity")
def get_popularity_index(ad_request: AdRequest):
    start=time.time()
    try:
        processed_data = preprocess(ad_request)
        popularity_index = prediction(processed_data)
        print(f"Took {time.time()-start} secs second process")
        return {"popularity_index": popularity_index, "second":time.time()-start}
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

if __name__ == "__main__":
    uvicorn.run("Popularity_api:app", port=8081, workers=1, timeout_keep_alive=10)

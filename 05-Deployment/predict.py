import pickle

# 1. Load the saved model
with open('pipeline_v1.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

# 2. Define the record to score
lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# 3. Transform the record using the DictVectorizer
X = dv.transform([lead])

# 4. Predict the probability of conversion
probability = model.predict_proba(X)[0, 1]

print(probability)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decimal import Decimal

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define base directory for loading dataset files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load and Train Model
try:
    df_comb = pd.read_csv(os.path.join(BASE_DIR, "dis_sym_dataset_comb.csv"))
    X = df_comb.iloc[:, 1:]
    Y = df_comb.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(x_train, y_train)

    rf_pred = random_forest.predict(x_test)
    acc_rf = round(Decimal(accuracy_score(y_test, rf_pred) * 100), 2)
    print(f"Model trained successfully. Accuracy: {acc_rf}%")

except Exception as e:
    print(f"Error loading dataset or training model: {e}")
    df_comb = None
    random_forest = None

# Load Doctors Data
try:
    df_doctors = pd.read_csv(os.path.join(BASE_DIR, "doctorsfinal_with_mobile.csv"))
except Exception as e:
    print(f"Error loading doctors data: {e}")
    df_doctors = None


def give_symptoms(disease):
    """Returns a list of symptoms for a given disease."""
    try:
        if df_comb is None:
            return []

        matching_rows = df_comb[df_comb.iloc[:, 0] == disease]

        if matching_rows.empty:
            return []

        all_symptoms = set()
        for _, row in matching_rows.iterrows():
            symptoms = row[1:][row[1:] == 1].index.tolist()
            all_symptoms.update(symptoms)

        return list(all_symptoms)
    except Exception as e:
        print(f"Error in give_symptoms: {e}")
        return []


def predict_top_5_diseases(symptoms):
    """Predicts the top 5 most likely diseases based on symptoms."""
    try:
        if random_forest is None:
            return []

        symptoms_dict = {symptom: 0.0 for symptom in X.columns}
        for symptom in symptoms:
            if symptom in symptoms_dict:
                symptoms_dict[symptom] = 1.0

        input_data = pd.DataFrame([symptoms_dict])

        probabilities = random_forest.predict_proba(input_data)
        top_indices = probabilities.argsort()[0][-5:][::-1]
        return [random_forest.classes_[i] for i in top_indices]

    except Exception as e:
        print(f"Error in predict_top_5_diseases: {e}")
        return []


def get_specialty_for_disease(disease):
    """Returns the medical specialty for a given disease."""
    try:
        df_specialty = pd.read_csv(os.path.join(BASE_DIR, "disease_speciality.csv"))

        specialty_row = df_specialty[df_specialty['Disease'].str.contains(disease, case=False, na=False)]
        if specialty_row.empty:
            return None

        return specialty_row.iloc[0]['Specialty']

    except Exception as e:
        print(f"Error in get_specialty_for_disease: {e}")
        return None


def get_doctors_for_disease(disease):
    """Returns a list of doctors specializing in a given disease."""
    try:
        if df_doctors is None:
            return []

        specialty = get_specialty_for_disease(disease)
        if not specialty:
            return []

        df_doctors["Specialty"] = df_doctors["Specialty"].str.strip().str.lower()
        specialty = specialty.strip().lower()

        matching_doctors = df_doctors[df_doctors["Specialty"].str.contains(specialty, case=False, na=False)]

        return [
            {
                "name": doctor["DoctorName"],
                "Specialty": doctor["Specialty"],
                "contact": doctor["IndianMobileNumber"],
                "city": doctor["City"],
            }
            for _, doctor in matching_doctors.iterrows()
        ]
    except Exception as e:
        print(f"Error in get_doctors_for_disease: {e}")
        return []


@app.route('/', methods=['POST'])
def index():
    """API endpoint to predict diseases based on symptoms."""
    try:
        data_json = request.get_json()
        symptoms = data_json.get("list", [])

        if not symptoms:
            return jsonify({"error": "Symptoms list cannot be empty."}), 400

        top_5_predictions = predict_top_5_diseases(symptoms)
        if not top_5_predictions:
            return jsonify({"error": "No prediction could be made."}), 500

        results = []
        for disease in top_5_predictions:
            all_symptoms = give_symptoms(disease)
            matched_symptoms = [symptom for symptom in symptoms if symptom in all_symptoms]

            doctor_info = get_doctors_for_disease(disease)

            results.append({
                "disease": disease,
                "all_symptoms": all_symptoms,
                "matched_symptoms": matched_symptoms,
                "doctors": doctor_info,
            })

        return jsonify(results)

    except Exception as e:
        print(f"Error in processing request: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use environment variable or default to 10000
    app.run(host="0.0.0.0", port=port)

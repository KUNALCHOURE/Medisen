from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decimal import Decimal

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

try:
    df_comb = pd.read_csv("dis_sym_dataset_comb.csv")
    X = df_comb.iloc[:, 1:]
    Y = df_comb.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(x_train, y_train)

    rf_pred = random_forest.predict(x_test)
    acc_rf = round(Decimal(accuracy_score(y_test, rf_pred) * 100), 2)
    print(f"Accuracy of Random Forest model: {acc_rf}%")

except Exception as e:
    print(f"Error loading data or training model: {e}")
    exit(1)

try:
    df_doctors = pd.read_csv("doctorsfinal_with_mobile.csv")
except Exception as e:
    print(f"Error loading doctors data: {e}")
    exit(1)


def give_symptoms(disease):
    try:
        matching_rows = df_comb[df_comb.iloc[:, 0] == disease]
        
        if matching_rows.empty:
            print(f"No data found for disease: {disease}")
            return []
        
        all_symptoms = []
        
        for _, row in matching_rows.iterrows():
            symptoms = row[1:][row[1:] == 1].index.tolist()
            all_symptoms.extend(symptoms)
        
        all_symptoms = list(set(all_symptoms))
        
        print(all_symptoms)
        return all_symptoms
    except Exception as e:
        print(f"Error in give_symptoms: {e}")
        return []


def predict_top_5_diseases(symptoms):
    try:
        symptoms_dict = {symptom: 0.0 for symptom in X.columns}
        
        for symptom in symptoms:
            if symptom in symptoms_dict:
                symptoms_dict[symptom] = 1.0
        
        input_data = pd.DataFrame([symptoms_dict])
        
        probabilities = random_forest.predict_proba(input_data)
        top_indices = probabilities.argsort()[0][-5:][::-1]
        top_diseases = [random_forest.classes_[i] for i in top_indices]

        return top_diseases
    except Exception as e:
        print(f"Error in predict_top_5_diseases: {e}")
        return []
    
def get_specialty_for_disease(disease):
    try:
      
        df_specialty = pd.read_csv("disease_speciality.csv") 
        
     
        specialty_row = df_specialty[df_specialty['Disease'].str.contains(disease, case=False, na=False)]
        
        if specialty_row.empty:
            print(f"No specialty found for disease: {disease}")
            return None
       
        specialty = specialty_row.iloc[0]['Specialty']
        return specialty
    
    except Exception as e:
        print(f"Error in get_specialty_for_disease: {e}")
        return None 
def get_doctors_for_disease(disease):
    try:
        # Get the required specialty for the disease
        specialty = get_specialty_for_disease(disease)
        print(f"Specialty for disease '{disease}': {specialty}")
        if not specialty:
            return [] 
        
        # Clean the specialty field in both the dataset and input
        df_doctors["Specialty"] = df_doctors["Specialty"].str.strip().str.lower()
        specialty = specialty.strip().lower()
        
        matching_doctors = df_doctors[
            df_doctors["Specialty"].str.contains(specialty, case=False, na=False)
        ]
        print(f"Matching doctors for specialty '{specialty}': {matching_doctors}")
        
        doctor_info = []
        for _, doctor in matching_doctors.iterrows():
            doctor_details = {
                "name": doctor["DoctorName"],
                "Specialty": doctor["Specialty"],
                "contact": doctor["IndianMobileNumber"],
                "city": doctor["City"]
            }
            doctor_info.append(doctor_details)
        
        return doctor_info
    except Exception as e:
        print(f"Error in get_doctors_for_disease: {e}")
        return []


@app.route('/', methods=['POST'])
def index():
    try:
        data_json = request.get_json()
        symptoms = data_json.get("list", [])
       
        print("Received symptoms:", symptoms)
      

        if not symptoms:
            print("Error: Symptoms list cannot be empty.")
            return jsonify({"error": "Symptoms list cannot be empty."}), 400

     
        
        top_5_predictions = predict_top_5_diseases(symptoms)

        if not top_5_predictions:
            print("No prediction made.")
            return jsonify({"error": "No prediction could be made."}), 500

        json_dict_list = []

        for disease in top_5_predictions:
            print(f"Processing disease: {disease}")
            
            all_symptoms = give_symptoms(disease)
            if not all_symptoms:
                print(f"No symptoms found for disease: {disease}")
                continue
            print(f"Predicted Disease: {disease}, Symptoms: {all_symptoms}")
            
            matched_symptoms = [symptom for symptom in symptoms if symptom in all_symptoms]
            if not matched_symptoms:
                print(f"No matching symptoms found for disease: {disease}")
            else:
                print(f"Matched Symptoms for disease {disease}: {matched_symptoms}")
            
            doctor_info = get_doctors_for_disease(disease)
            if not doctor_info:
                print(f"No doctors found for disease: {disease}")
            else:
                for doctor in doctor_info:
                    print(f"Name: {doctor['name']}, Speciality: {doctor['Specialty']}, Contact: {doctor['contact']}, City: {doctor['city']}")
            
            json_dict = {
                "disease": disease,
                "all_symptoms": all_symptoms,
                "matched_symptoms": matched_symptoms,
                "doctors": doctor_info
            }
            json_dict_list.append(json_dict)

        if not json_dict_list:
            print("No predictions or doctor information available.")
        else:
            print("Final JSON Response:", json_dict_list)

        return jsonify(json_dict_list)

    except Exception as e:
        print("Error in processing:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Change the port if needed


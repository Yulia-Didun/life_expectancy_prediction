import pickle
import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

model = joblib.load('model/model.pkl')
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

data = pd.read_csv('life_expectancy_preprocessed.csv')
country_list = sorted(data['Country'].unique())

smoking_map = {'none': 0.0, 'light': 0.5, 'regular': 0.3}

def normalize_income(country_data, income_level):
    avg_income_composition_series = country_data['Income_Composition_Of_Resources']

    if avg_income_composition_series.empty:
        avg_income_composition = 0.7
    else:
        avg_income_composition = avg_income_composition_series.iloc[0]
        print(avg_income_composition)
    
    income_levels = {
        'very_low': avg_income_composition * 0.45,
        'low': avg_income_composition * 0.75,
        'medium': avg_income_composition,
        'high': min(avg_income_composition * 1.25, 0.90),
        'very_high': min(avg_income_composition * 1.55, 0.95),
    }
    print(avg_income_composition)
    
    return income_levels.get(income_level, avg_income_composition)

def generate_health_advice(bmi, smoking, alcohol_servings_per_week):
    factors = []
    recommendations = []

    if bmi < 18.5:
        factors.append({
            'type': 'negative',
            'title': 'Low BMI',
            'description': 'Being underweight can increase risk of health problems.'
        })
        recommendations.append({
            'icon': '🥗',
            'title': 'Maintain a Healthy Weight',
            'description': 'Consider a nutritious diet to reach a healthy weight.'
        })
    elif 18.5 <= bmi <= 24.9:
        factors.append({
            'type': 'positive',
            'title': 'Healthy BMI',
            'description': 'Your weight is within a healthy range.'
        })
    else:
        factors.append({
            'type': 'negative',
            'title': 'High BMI',
            'description': 'Overweight or obesity increase risk of chronic diseases.'
        })
        recommendations.append({
            'icon': '🥗',
            'title': 'Manage Your Weight',
            'description': 'Try to maintain a balanced diet and regular exercise.'
        })

    if smoking == 'none':
        factors.append({
            'type': 'positive',
            'title': 'Non-Smoker',
            'description': 'Not smoking reduces risks of many diseases.'
        })
    else:
        factors.append({
            'type': 'negative',
            'title': 'Smoking',
            'description': 'Smoking significantly increases risk of cancer and heart disease.'
        })
        recommendations.append({
            'icon': '🚭',
            'title': 'Quit Smoking',
            'description': 'Quitting smoking greatly improves your health and life expectancy.'
        })

    if alcohol_servings_per_week <= 7:  # приблизно 1 порція в день
        factors.append({
            'type': 'positive',
            'title': 'Moderate Alcohol Consumption',
            'description': 'Alcohol intake is within recommended limits.'
        })
    else:
        factors.append({
            'type': 'negative',
            'title': 'High Alcohol Consumption',
            'description': 'Excessive alcohol consumption increases health risks.'
        })
        recommendations.append({
            'icon': '🍷',
            'title': 'Reduce Alcohol Intake',
            'description': 'Limit alcohol consumption to recommended levels.'
        })

    recommendations.append({
        'icon': '🏃',
        'title': 'Increase Physical Activity',
        'description': 'Aim for at least 150 minutes of moderate exercise per week.'
    })

    recommendations.append({
        'icon': '🛌',
        'title': 'Get Enough Sleep',
        'description': 'Aim for 7-9 hours of quality sleep every night.'
    })

    return factors, recommendations

@app.route('/')
def index():
    countries = sorted(data['Country'].unique())
    return render_template('life-expectancy-calculator.html', countries=countries)

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    age = int(data_json['age'])
    country = data_json['country']
    height_cm = float(data_json['height'])
    weight_kg = float(data_json['weight'])
    smoking = data_json['smoking']
    alcohol_servings_per_week = float(data_json['alcohol'])
    schooling = float(data_json['education'])
    income_level = data_json['income']

    bmi = weight_kg / ((height_cm / 100) ** 2)

    country_data = data[(data['Country'] == country) & (data['Year'] == 2015)]
    country_smoking_rate = country_data['Smoking_Rate']

    smoking_factor = smoking_map.get(smoking, 0)
    user_smoking_score = country_smoking_rate * smoking_factor

    alcohol_litres_per_year = alcohol_servings_per_week * 0.0158 * 52

    normalized_income = normalize_income(country_data, income_level)

    input_data = pd.DataFrame({
        'Year': [2015],
        'Status': [country_data['Status']],
        'Adult_Mortality': [country_data['Adult_Mortality']],
        'Alcohol': [alcohol_litres_per_year],
        'Hepatitis_B': [country_data['Hepatitis_B']],
        'Measles': [country_data['Measles']],
        'BMI': [bmi],
        'Under-Five_Deaths': [country_data['Under-Five_Deaths']],
        'Polio': [country_data['Polio']],
        'Total_Expenditure': [country_data['Total_Expenditure']],
        'Diphtheria': [country_data['Diphtheria']],
        'HIV/AIDS': [country_data['HIV/AIDS']],
        'GDP': [country_data['GDP']],
        'Population': [country_data['Population']],
        'Thinness__1-19_Years': [country_data['Thinness__1-19_Years']],
        'Income_Composition_Of_Resources': [normalized_income],
        'Schooling': [schooling],
        'Smoking_Rate': [user_smoking_score],
    })

    input_scaled = scaler.transform(input_data)
    predicted_life_expectancy = model.predict(input_scaled)[0]

    factors, recommendations = generate_health_advice(bmi, smoking, alcohol_servings_per_week)

    return jsonify(result=round(predicted_life_expectancy, 1),
                   factors=factors,
                   recommendations=recommendations
                   )

@app.route('/result')
def result_page():
    predicted_life_expectancy = request.args.get('predicted_life_expectancy', type=float)
    age = request.args.get('age', type=int)

    if predicted_life_expectancy and age:
        years_left = predicted_life_expectancy - age
        percent_lived = (age / predicted_life_expectancy) * 100 if predicted_life_expectancy > 0 else 0

        return render_template('result.html',
                               predicted_life_expectancy=round(predicted_life_expectancy, 1),
                               age=age,
                               years_left=round(years_left, 1),
                               percent_lived=round(percent_lived, 1))
    else:
        return "Missing parameters", 400

if __name__ == "__main__":
    app.run(debug=True)

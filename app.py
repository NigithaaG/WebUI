from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('xgb_model_top_10.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    follicle_no_r = float(request.form['follicle_no_r'])
    follicle_no_l = float(request.form['follicle_no_l'])
    hair_growth = 1 if request.form['hair_growth'].lower() == 'y' else 0
    skin_darkening = 1 if request.form['skin_darkening'].lower() == 'y' else 0
    weight_gain = 1 if request.form['weight_gain'].lower() == 'y' else 0
    cycle = 1 if request.form['cycle'].lower() == 'r' else 0
    lh = float(request.form['lh'])
    fast_food = 1 if request.form['fast_food'].lower() == 'y' else 0
    fsh_lh = float(request.form['fsh_lh'])
    cycle_length = float(request.form['cycle_length'])

    features = np.array([[follicle_no_r, follicle_no_l, hair_growth, skin_darkening, weight_gain,
                          cycle, lh, fast_food, fsh_lh, cycle_length]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    result = "Yes" if prediction == 1 else "No"
    percentage = probability[1] * 100

    return render_template('result.html', result=result, percentage=percentage)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/risk_level')
def risk_level():
    return render_template('form.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/track')
def track():
    return render_template('tracker.html')

if __name__ == '__main__':
    app.run(debug=True)

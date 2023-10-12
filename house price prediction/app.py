from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle
app = Flask(__name__)
def ValuePredictor(to_predict_list):
    X_test = np.array(to_predict_list).reshape(1, 1)
    #Load the instance of Standarscalar object
    scaler = pickle.load(open("scaler.pkl", "rb"))
    #Normalize the data
    X_test_Normalized = scaler.transform(X_test)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(X_test_Normalized)
    return result[0]
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        prediction = ValuePredictor(to_predict_list)
    return render_template("result.html", prediction = prediction)
@app.route("/")
def hello_world():
    return render_template("home.html")
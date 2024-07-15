from flask import Flask, redirect, url_for, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    data = json_[0]
    output = []
    for i in range(len(json_)):
        prediction = loaded_model.predict(np.array(list(json_[i].values())).reshape(1, 12))
        output.append(int((int(prediction[0]) == 1)))
    return jsonify(Predictions = output, status = 200, mimetype = 'application/json')

@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = "Income more than 50K"
        else:
            prediction = "Income less that 50K"
        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run()

import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request
import pickle

# model
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods=['POST'])

def make_predict():
    #get data
    data = request.get_json(force=True)

    predict_request = [data['neighborhood'],
                       data['room_type'],
                       data['accommodates'],
                       data['bedrooms'],
                       data['number_of_reviews'],
                       data['wifi'],
                       data['cable_tv'],
                       data['washer'],
                       data['kitchen']]

    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # preds
    y_hat = model.predict(data_df) # this works for xgb

    # send back to browser
    output = {'y_hat': int(y_hat[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 9000, debug=True)


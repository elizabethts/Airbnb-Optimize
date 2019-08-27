import numpy as np
from flask import Flask, jsonify, request
import pickle

# model
# my_model = pickle.load(open('/tuesday_model_6.pkl','rb'))
my_model = pickle.load(open('tuesday_model_6.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods=['POST'])

def make_predict():
    #get data
    data = request.get_json(force=True)

    # testing on tuesday model 6
    predict_request = [data['neighborhood'],
                       data['room_type'],
                       data['accommodates'],
                       data['bedrooms'],
                       data['bathrooms'],
                       data['number_of_reviews'],
                       data['wifi'],
                       data['cable_tv'],
                       data['washer'],
                       data['kitchen']]


    predict_request = np.array(predict_request).reshape(1,-1)

    #preds
    y_hat = my_model.predict(predict_request)

    # send back to browser
    output = {'y_hat': int(y_hat[0])}
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port = 5000, debug=True)

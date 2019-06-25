import os
import json
import pickle
import flask
import pandas as pd

prefix = "dir/"
model_path = os.path.join(prefix, 'model')

app = flask.Flask(__name__)
debug = True
app.debug = debug


class ScoringService:
    """Class that loads the model in memory."""
    model = None  # the model itself

    @classmethod
    def get_model(cls):
        """Loads the model"""
        if cls.model is None:
            with open(os.path.join(model_path, 'great_model.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """Predicts for the passed data"""
        # load model
        model = cls.get_model()
        # get the data
        data = pd.DataFrame(input)
        # now we can do the inference
        return model.predict(data)


@app.route('/ping', methods=['GET'])
def ping():
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Bienvenido a la API de detección de anomalías.</h1>
<p>Algoritmo: Isolation Forest.</p>'''


@app.route('/ad', methods=['POST'])
def api_predict():
    data = None
    if flask.request.method == 'POST':
        data = json.loads(flask.request.data)
    if data is None:
        return flask.jsonify("The transformation didn't succeed.")

    # Do the prediction
    predictions = ScoringService.predict(data)

    # transform to json
    predictions = flask.jsonify(predictions.tolist())

    return predictions


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=debug)

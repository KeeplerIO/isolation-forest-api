import pickle
import sys
import os
import json
import traceback
import pandas as pd
from pyod.models.iforest import IForest


# directory where all the info is stored
prefix = "dir/"

# ==== some paths needed
input_path = prefix + 'input/data'  # data input path
output_path = os.path.join(prefix, 'output')  # output path
model_path = os.path.join(prefix, 'model')  # path to save the model
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')  # configuration
training_path = os.path.join(input_path, 'train/X.csv')


def train():
    # initiate the train
    try:
        # 1-. Take data and configuration
        data = pd.read_csv(training_path, index_col=0)

        # Read in any configuration stored
        with open(param_path, 'r') as tc:
            hyper_parameters = json.load(tc)

        # 2-. Set up
        # instantiate the Isolation Forest model
        model = IForest(contamination=hyper_parameters['contamination'],
                        behaviour='new')
        model.fit(data)  # fit

        # 3-. Save the model
        model_name = 'great_model'
        with open(os.path.join(model_path, '{}.pkl'.format(model_name)), 'wb') as out:
            pickle.dump(model, out, protocol=0)

    # consider that the train fails
    except Exception as e:
        # write the log
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during train: ' + str(e) + '\n' + trc)
        sys.exit(255)


if __name__ == '__main__':
    train()


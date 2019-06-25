import pandas as pd
import requests
import json

# API url
url = "http://0.0.0.0:5000/ad"
# data path
data = "dir/input/data/test/X.csv"
# load data to test the API - we are not testing the model!
x = pd.read_csv(data, index_col=0)
# make the post
print(requests.post(url, data=json.dumps(x.to_dict())).content)

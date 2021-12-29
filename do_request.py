import json

import pandas as pd
import requests

while True:
    inp = input()
    if inp == 'q':
        break

    data = {'Text': [inp]}
    data = pd.DataFrame(data=data)

    request = requests.post("http://localhost:5000/predict", data.to_json())
    print(request.status_code)

    response = json.loads(request.content)
    response_df = pd.DataFrame(response)
    print(f'response: {response["Prediction"]["0"]}')

from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__) # create the Flask app
CORS(app) # allow requests from other domains

@app.route('/predict', methods=['POST'])
def predict():
    url = request.get_json()
    loaded_model = joblib.load(open('./knn.pkl',
                                    'rb'))

    para = pd.DataFrame(columns=( 'Epoch', 'HR'))
    para = para.append(pd.DataFrame({'Epoch':[url["Epoch"]],'HR':[url["HR"]]}),ignore_index=True)
    result = loaded_model.predict(para)

    return str(result[0])


    ##urllib.request.urlretrieve(url, 'image.bmp')


if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000

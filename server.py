import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from flask import jsonify
from io import BytesIO
import base64
import re

app = Flask(__name__)

# test data
tpe = {
    "id": 0,
    "city_name": "Taipei",
    "country_name": "Taiwan",
    "is_capital": True,
    "location": {
        "longitude": 121.569649,
        "latitude": 25.036786
    }
}
nyc = {
    "id": 1,
    "city_name": "New York",
    "country_name": "United States",
    "is_capital": False,
    "location": {
        "longitude": -74.004364,
        "latitude": 40.710405
    }
}
ldn = {
    "id": 2,
    "city_name": "London",
    "country_name": "United Kingdom",
    "is_capital": True,
    "location": {
        "longitude": -0.114089,
        "latitude": 51.507497
    }
}
cities = [tpe, nyc, ldn]

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
cardNumber = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    
    filenamestr = feature_path.stem+".jpg"
    cardfirst = filenamestr[0:1]
    cardsecond = filenamestr[0:filenamestr.rfind('_')]
    cardthird = filenamestr[0:filenamestr.rfind('.')]
    img_paths.append("https://ws-tcg.com/wordpress/wp-content/images/cardlist/"+cardfirst+"/"+cardsecond+"/"+cardthird+".png")
    cardNumber.append(feature_path.stem)
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        b64Full = request.values.get('imgimg')
        b64 = re.sub('data:image\/jpeg;base64,','',b64Full)
        print(b64)
        img = Image.open(BytesIO(base64.b64decode(b64)))
        # Save query image
        #img = Image.open(file.stream)  # PIL image <-
        #img = img.thumbnail((600, 600))
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        #img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:5]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        cardNumberList = [(cardNumber[id]) for id in ids]
        for id in ids:
            print("id:"+id)
            print(cardNumber[id])
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores,
                               cardNumberList=cardNumberList)
    else:
        return render_template('index.html')
 

@app.route('/rese', methods=['GET', 'POST'])
def rese():

    name = request.args.get('name')
    fruit = request.args.get('fruit')
    response = jsonify({'name': name},{'fruit':fruit}) 
    #return render_template('rese.html',**locals()
    return response


if __name__=="__main__":
    app.run("0.0.0.0")

from flask import Flask, render_template, request
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)
model = pickle.load(open('mobile_price_prediction.pkl', 'rb'))

#url
#app.route("/", methods=['GET','POST'])

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        feat1=int(request.form['feat1'])
        feat2=int(request.form['feat2'])
        feat3=int(request.form['feat3'])
        feat4=int(request.form['feat4'])
        feat5=int(request.form['feat5'])
        feat6=int(request.form['feat6'])
        feature=np.array([[feat1],[feat2],[feat3],[feat4],[feat5],[feat6]])
        scale=MinMaxScaler()
        feature_scale=scale.fit_transform(feature)
        feature_reshape=feature_scale.reshape(1,6)

        prediction=model.predict(feature_reshape)
        if prediction==0:
            return render_template('index.html',names='Phone is cheap')
        if prediction==1:
            return render_template('index.html',names='Phone is medium rate')
        if prediction==2:
            return render_template('index.html',names='Phone is expensive')
        if prediction==3:
            return render_template('index.html',names='Phone is very expensive')

    return render_template('index.html')

#@app.route("/output.html")
# def output():
#     return render_template('output.houtput

if __name__=='__main__':
    #just to reload the server by itself when we made any changes
    app.run(debug=True)

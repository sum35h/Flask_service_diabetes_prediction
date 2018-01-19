from flask import Flask,request, jsonify
from flask_restful import Resource,Api
import pandas as pd
from sklearn.linear_model import LogisticRegression
app=Flask(__name__)
api=Api(app)
@app.route('/home', methods=['GET'])
def home():
     return ("Prediction Service is Running!")

@app.route('/predict', methods=['POST'])
def perdict():
    #gets festures from user request payload
    user_data=request.get_json()
    df=pd.DataFrame(user_data);


    dataset = pd.read_csv('diabetes.csv')
    #target var
    outcome=dataset['Outcome']
    #features
    data=dataset[dataset.columns[:8]]

    #logistic regression model
    model = LogisticRegression()
    #train
    model.fit(data,outcome)
    #predict
    prediction=model.predict(df).tolist();# predict() returns a list of int64
    #tolist() converts int64 list to python native int
    #int64 cant be converted to json ...got some error

    #return the prediction as json
    return jsonify({'prediction': prediction})


app.run()

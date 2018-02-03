import time
from flask import Flask,request, jsonify
from flask_restful import Resource,Api
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
app=Flask(__name__)
api=Api(app)
global model

@app.route('/home', methods=['GET'])
def home():
     return ("Prediction Service is Running!")

@app.route('/predict', methods=['POST'])
def perdict():
    #gets festures from user request payload
    if model:
        user_data=request.get_json()
        df=pd.DataFrame(user_data);
        print("in")

        dataset = pd.read_csv('diabetes.csv')
        #target var
        #outcome=dataset['Outcome']
        #features
        #data=dataset[dataset.columns[:8]]

        #logistic regression model
        #model = LogisticRegression()
        #train
        #model.fit(data,outcome)
        #predict
        prediction=model.predict(df).tolist();# predict() returns a list of int64
        #tolist() converts int64 list to python native int list
        #int64 cant be converted to json ...got some error

        #return the prediction as json
        return jsonify({'prediction': prediction})
    else:
        return 'no model, train first'




model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory


@app.route('/train', methods=['GET'])
def train():

        dataset = pd.read_csv('diabetes.csv')
        #target var
        outcome=dataset['Outcome']
        #features
        data=dataset[dataset.columns[:8]]

        #logistic regression model
        model = LogisticRegression()
        #train
        start = time.time()
        model.fit(data,outcome)
        #predict



        print ("Trained in %.1f seconds" ,(time.time() - start))
        print ("Model training score: %s" , model.score(data,outcome))

        joblib.dump(model, model_file_name)

        return 'Success'


if __name__ == '__main__':


    try:
        model = joblib.load(model_file_name)
        print( 'model loaded==============================================================')


    except :
        print ('No model here,Train first')
        model=None


app.run()

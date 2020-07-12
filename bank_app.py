import os
os.chdir("E:\Data Science\Projects\Bank_Churn_Prediction")
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import keras
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Activation, Dense
app = Flask(__name__)
mn= pickle.load(open('mn.pkl','rb'))
mn
model = load_model('bank_churn.h5')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
     if request.method == 'POST':
        CreditScore= float(request.form['CreditScore'])
        Age=int(request.form['Age'])
        Tenure = int(request.form['Tenure'])
        Balance=float(request.form['Balance'])
        NumOfProducts=int(request.form['NumOfProducts']) 
        EstimatedSalary=float(request.form['EstimatedSalary'])
        HasCreditCard=request.form['HasCreditCard']
        if(HasCreditCard=='Yes'):
            HasCrCard=1
        else:
            HasCrCard=0
        ActiveMember=request.form['ActiveMember']
        if(ActiveMember=='Yes'):
            IsActiveMember=1
        else:
            IsActiveMember=0  
        Geography=request.form['Geography']
        if(Geography=='Germany'):
            Geography_Germany=1
            Geography_Spain=0
        elif(Geography=='Spain'):
            Geography_Germany=0
            Geography_Spain=1
        else:
            Geography_Germany=0
            Geography_Spain=0
        Gender=request.form['Gender']
        if(Gender=='Male'):
            Gender_Male=1
        else:
            Gender_Male=0
        c = mn.transform([[CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography_Germany,Geography_Spain,Gender_Male]])
        m_prediction = model.predict(c)
        m_prediction = m_prediction>0.5
        return render_template('result.html', prediction=m_prediction)
if __name__ == '__main__':
	app.run(debug=True)
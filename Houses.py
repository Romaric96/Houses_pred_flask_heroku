from flask import Flask,render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split


app=Flask(__name__)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

@app.route("/", methods=['POST','GET'])
def index():
    note = ''
    results=[]
    if request.method == 'POST':
        OverallQual = request.form['OverallQual']
        GrLivArea = request.form['GrLivArea']
        GarageCars = request.form['GarageCars']
        GarageArea = request.form['GarageArea']
        TotalBsmtSF = request.form['TotalBsmtSF']
        FstFlrSF = request.form['FstFlrSF']
        FullBath = request.form['FullBath']
        TotRmsAbvGrd = request.form['TotRmsAbvGrd']
        YearBuilt = request.form['YearBuilt']
        YearRemodAdd = request.form['YearRemodAdd']
        MasVnrArea = request.form['MasVnrArea']
        GarageYrBlt = request.form['GarageYrBlt']
        Fireplaces = request.form['Fireplaces']
        BsmtFinSF1 = request.form['BsmtFinSF1']

        X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.3, random_state=150)
        y_train= y_train.values.reshape(-1,1)
        y_test= y_test.values.reshape(-1,1)

        model = joblib.load('gradientboosting.pkl')
        data = [[OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,FstFlrSF,FullBath,TotRmsAbvGrd,YearBuilt,YearRemodAdd,MasVnrArea,GarageYrBlt,Fireplaces,BsmtFinSF1]]
        results = [OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,FstFlrSF,FullBath,TotRmsAbvGrd,YearBuilt,YearRemodAdd,MasVnrArea,GarageYrBlt,Fireplaces,BsmtFinSF1]

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(data)
        y_test = sc_y.fit_transform(y_test)

        predict =  model.predict(X_train)
        predict = predict.reshape(-1,1)
        predict = sc_y.inverse_transform(predict)
        note = predict

        print(note)
    return render_template ('index.html', note=note, results=results)


if __name__=='__main__':
    app.run(debug=True)
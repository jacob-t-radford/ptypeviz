from flask import render_template,Flask,request,jsonify
from app import app
import joblib
import numpy as np
import tensorflow as tf

# Standard scaler generated from training data
scaler = joblib.load("./app/static/height_scaler.save")

# 1DCNN Height AGL model
mymodel = tf.keras.models.load_model("./app/static/height_model.h5")

@app.route('/',methods=['GET','POST'])
@app.route('/index',methods=['GET','POST'])
def index():
        return render_template('index.html')


# Retrieve sounding data via ajax, transform, then predict ptype using CNN and return via ajax
@app.route('/loadSounding',methods=['GET','POST'])
def loadSounding():
        myjson = request.get_json()
        temp = myjson["temp"][::-1]
        dew = myjson["dew"][::-1]
        uwnd = myjson["uwnd"][::-1]
        vwnd = myjson["vwnd"][::-1]
        mydata = temp + dew + uwnd + vwnd
        mydata = np.array(mydata).reshape(1,-1)
        mydatatransformed = scaler.transform(mydata)
        mydatatransformed = mydatatransformed.reshape((1,67,4),order='F')
        mylabel = mymodel.predict(mydatatransformed)
        mylabellist = mylabel[0].tolist()
        mylabeljson = jsonify({'labels':mylabellist})
        return(mylabeljson)

# Retrieve modified sounding data via ajax, transform, then predict new ptype and return via ajax
@app.route('/adjustSounding',methods=['GET','POST'])
def adjustSounding():
        mynewdata = request.get_json()
        mynewdata = np.array(mynewdata)
        mynewdata = mynewdata.reshape(4,67)
        mynewdata = np.flip(mynewdata,axis=1)
        mynewdata = mynewdata.reshape(1,-1)
        mynewdatatransformed = scaler.transform(mynewdata)
        mynewdatatransformed = mynewdatatransformed.reshape((1,67,4),order='F')
        mynewlabel = mymodel.predict(mynewdatatransformed)
        mynewlabellist = mynewlabel[0].tolist()
        mynewlabeljson = jsonify({'newlabels':mynewlabellist})
        return(mynewlabeljson)

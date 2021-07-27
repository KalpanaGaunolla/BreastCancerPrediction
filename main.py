from flask import Flask,request,render_template,redirect,url_for
import pickle,gzip
import joblib
import numpy as np
model = joblib.load('BreastCancer.pkl')

app=Flask(__name__)
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
        Age=float(request.form.get('Age',False))
        BMI= float(request.form.get('BMI',False))
        Adiponectin= float(request.form.get('Adiponectin',False))
        Glucose= float(request.form.get('Glucose',False))
        Insulin=float(request.form.get('Insulin',False))
        Resistin= float(request.form.get('Resistin',False))

        arr=np.array([[Age,BMI,Adiponectin,Glucose,Insulin,Resistin]])
        pred=model.predict(arr)
        if pred==1:
            res_Val="Healthy controls"
        else:
            res_Val="Risk of breast cancer"

        return render_template('index.html',prediction_text='You are having {}'.format(res_Val))


if __name__=='__main__':
    app.run(debug=True)




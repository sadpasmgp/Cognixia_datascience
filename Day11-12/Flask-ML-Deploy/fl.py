import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    #return "Hello World!"

@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)

@app.route('/predict',methods=['POST'])
def predict():
    int_sal = [int(x) for x in request.form.values()]
    final_sal = [np.array(int_sal)]
    prediction = model.predict(final_sal)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Salary should be $ {}'.format(output))

if __name__ == '__main__':
    app.run()

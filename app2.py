# PREDICTION OF IMAGE MODEL
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


from flask import Flask, request, jsonify
import pickle
import joblib
from sklearn.svm import SVC
import os
# model2 = SVC(probability=True)
# Train the model and save it
# joblib.dump(model2, 'svc_model.pkl')

model = joblib.load('svm_model2.pkl')
@app.route('/predict4', methods=['POST'])
def predict4():
    data = request.json
    print(data)
    a1 = [ 0.30669145,  0.27509294,  0.37918216,  2.55576208,  4.93866171,
        0.49070632, 24.31189591,  0.4535316 , 33.8401487 , 37.99814126,
       59.64405204, 31.42007435]
    a2 = [ 0.46112016,  0.4465611 ,  0.48518352,  0.89901551,  1.49159718,
        0.49991362,  4.0604411 ,  0.497836  ,  3.59994134,  3.97272255,
       11.04075805,  5.40876736]
    
    # for i in range(12):
    #     data[i] = (data[i]-a1[i])/a2[i]
    # data[0]=(data[0]-31.42007435)/5.40876736

    lol = []
    lol2 = []
    d = data.get('input')
    print("d1=" , d)
    for i in range(12):
        lol.append(float(d[i]))
    lol2.append(lol)
    print("d=" , lol2)

    for i in range(12):
        lol2[0][i] = (lol2[0][i]-a1[i])/a2[i]
    
    print(lol2)

    def rbf_kernel(x, y, gamma):
        return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

    # support_vectors = model.support_vectors_
    # dual_coefficients = model.dual_coef_[0]
    # intercept = model.intercept_[0]
    # gamma = model._gamma

    # def predict(sample, support_vectors, dual_coefficients, intercept, gamma):
    #     decision = sum(dual_coefficients[i] * rbf_kernel(sample, support_vectors[i], gamma) for i in range(len(support_vectors)))
    #     # print(decision, intercept)
    #     # return np.sign(decision + intercept)
    #     return 1 if (decision + intercept)>0 else 0
    
    xsamp = lol2
    # pred1 = predict(xsamp, support_vectors, dual_coefficients, intercept, gamma)
    # print(pred1)
    probb = model.predict_proba(xsamp)
    s = round(probb[0][0],2)
    # prediction = model.predict(lol2)
    # print(prediction)
    # s = prediction

    
    return jsonify({'prediction': s})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv("PORT", 5000))


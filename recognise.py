import numpy as np

from keras.models import Sequential, load_model
from sklearn.preprocessing import OneHotEncoder
model = load_model('cnn_classifier.h5')
def recognise(cropped):
    d = np.reshape(cropped, (1, 28, 28, 1))
    out = model.predict(d)
    # Get max pre arg
    p = []
    precision = 0
    for i in range(len(out)):
        z = np.zeros(36)
        z[np.argmax(out[i])] = 1.
        precision = max(out[i])
        p.append(z)
    prediction = np.array(p)

    # Inverse one hot encoding
    alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    classes = []    
    for a in alphabets:
        classes.append([a])
    ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
    ohe.fit(classes)
    pred = ohe.inverse_transform(prediction)

    if precision > 0.6:
        print('Prediction : ' + str(pred[0][0]) + ' , high : ' + str(precision))
        return pred[0][0]
    else:
        print(('Prediction : ' + str(pred[0][0]) + ' , Low : ' + str(precision)))
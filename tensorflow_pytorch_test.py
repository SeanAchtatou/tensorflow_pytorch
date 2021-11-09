import numpy as np
import pandas as pd
import tensorflow as tf
import torch as tc

from keras import Sequential, layers, metrics
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
from aix360.datasets.heloc_dataset import HELOCDataset, nan_preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from art.attacks.evasion import SaliencyMapMethod
from torch import nn


##################################################### DATA AND PREPROCESS ##############################################

data = HELOCDataset(custom_preprocessing=nan_preprocessing).data()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
print("Size of HELOC dataset:", data.shape)
print("Number of \"Good\" applicants:", np.sum(data['RiskPerformance']==1))
print("Number of \"Bad\" applicants:", np.sum(data['RiskPerformance']==0))

labels = data.pop('RiskPerformance')              #GET THE LABELS

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0, stratify=labels)
X_train = np.nan_to_num(X_train)                                 #REPLACE MISSING VALUES OR TOO HIGH
X_test = np.nan_to_num(X_test)

sc = StandardScaler()                                            #STANDARIZE THE VALUES
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

############################################################################ MODELS CREATION TENSORFLOW ##########################

model = Sequential()
model.add(Dense(50, input_dim=23))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.01)
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

plot_h = model.fit(X_train, y_train, epochs=50, batch_size=256)



plt.plot(plot_h.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train model'], loc='upper left')
plt.show()

plt.plot(plot_h.history['loss'])
plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')
plt.show()

y_prob_tensor = model.predict(X_test)
y_pred_tensor = np.argmax(y_prob_tensor,axis=1)
#y_evaluate = model.evaluate(X_test,y_test)
#print(f"Accuracy on test : {y_evaluate}")

#print(y_prob_tensor)
#print(y_pred_tensor)

clf = RandomForestClassifier(n_estimators=500, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

#print(y_proba)
#print(y_pred)


print('Mean Absolute Error SKLEARN:', metrics.mean_absolute_error(y_test, y_pred))
print('ROC AUC Error SKLEARN:', metrics.roc_auc_score(y_test, y_proba[:,1]))
print('Mean Absolute Error TENSORFLOW:', metrics.mean_absolute_error(y_test, y_pred_tensor))
print('ROC AUC Error TENSORFLOW:', metrics.roc_auc_score(y_test, y_prob_tensor))

################################################## JACOBIAN SALIENCY MAP ATTACK ########################################

# Issue we have a DNN, it attends to obtain a trained classifier. Obtaining following error : art.exceptions.EstimatorError: SaliencyMapMethod requires an estimator derived from <class 'art.estimators.estimator.BaseEstimator'> and <class 'art.estimators.classification.classifier.ClassGradientsMixin'>, the provided classifier is an instance of <class 'keras.engine.sequential.Sequential'> and is derived from (<class 'keras.engine.functional.Functional'>,).
JSMA_ModelAttack = SaliencyMapMethod(classifier=model)
adversial_100 = []
for i in range(100):
     adversial_100.append(JSMA_ModelAttack.generate(X_train))

#y_prob_real = model.predict(X_test)
#y_prob_attack = model.predict(adversial_100[0])
#y_pred_real = np.argmax(y_prob_real,axis=1)
#y_pred_attack = np.argmax(y_prob_attack,axis=1)

y_evaluate = model.evaluate(X_test,y_test)
print(f"Accuracy on test real : {y_evaluate}")

y_evaluate_attack = model.evaluate(adversial_100[0],y_test)
print(f"Accuracy on test attack : {y_evaluate}")



################################################################ MODEL CREATION PYTORCH ########################################################

model = nn.Sequential(nn.Linear(23, 50),
                      nn.ReLU(),
                      nn.Linear(50, 80),
                      nn.ReLU(),
                      nn.Linear(80, 100),
                      nn.ReLU(),
                      nn.Linear(100, 80),
                      nn.ReLU(),
                      nn.Linear(80, 1),
                      nn.Sigmoid())


X_train = tc.tensor(X_train)
criterion = nn.CrossEntropyLoss()
optimizer = tc.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(50):
    running_loss=0.0
    for i in range(256):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train[i])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
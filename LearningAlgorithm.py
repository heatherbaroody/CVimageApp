#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:55:40 2018

@author: heather
"""
#Import packages

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from PIL import Image as IMG

# In[1] Process Data Library 

files = []
num111 = 8
num110 = 6
num100 = 6
numpoly = 7
l1 = ['111','110','100','poly']
l2=[num111, num110, num100, numpoly]
for j in range(len(l1)):
    for i in range(l2[j]):
        files.append("pt{}-{}.jpg".format(l1[j],i+1))

images = []
for file in files:
    images.append(IMG.open(file).convert('I'))

#resize image files
x_pix = 120
y_pix = 120
image_rs = []

for i in images:
    image_rs.append(i.resize((x_pix, y_pix)))

#convert to array
image_ar = []
for j in image_rs:
    image_ar.append(np.array(j))

image_ar = np.array(image_ar)


# resize matrix for use in algorithm
dim = np.dot(image_ar.shape[1],image_ar.shape[2])
data = np.reshape(image_ar, (image_ar.shape[0], dim))
data.shape


# In[2] Label Data

#Define target values
Pt111_index = np.zeros((1,num111))
Pt110_index = np.ones((1,num110))
Pt100_index = 2*np.ones((1,num100))
Ptpoly_index = 3*np.ones((1,numpoly))

target = np.concatenate((Pt111_index,Pt110_index,Pt100_index,Ptpoly_index),axis=1)
target = target.reshape(data.shape[0],)

facet = ['Pt (111)', 'Pt(110)', 'Pt(100)', 'Pt(poly)']


# In[7]: Split data into training set and validation set


Xtrain, Xtest, ytrain, ytest = train_test_split(data, target,
                                                random_state=2)
print(Xtrain.shape, Xtest.shape)


# In[8]:Training algorithm - Logistic Regression

clf = LogisticRegression(penalty='l2')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

# In[8] TEST FUNCTION
# Process test image
#cvplot = "pt100-test.jpg"

def analyzeCV(cvplot):
    img = IMG.open(cvplot).convert('I')
    test_img = img.resize((120, 120))
    test = np.array(test_img)
    dimt = np.dot(test.shape[0],test.shape[1])
    test = np.reshape(test, (1, dimt))
    #Run test
    usertest = clf.predict(test)
    usertest = np.int(usertest)
    #Print test
    print(facet[usertest])
    return facet[usertest]

#Train KNN Classifier
## create the model
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
# fit the model
knn.fit(data, target)

## Probablities function
def KNNpredict(cvplot):
    img = IMG.open(cvplot).convert('I')
    test_img = img.resize((120, 120))
    test = np.array(test_img)
    dimt = np.dot(test.shape[0],test.shape[1])
    test = np.reshape(test, (1, dimt))
    ## create the model
    #knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    # fit the model
    #knn.fit(data, target)
    #test = np.array(test)
    #test.shape
    #usertest = clf.predict()
    result = knn.predict(test)
    KNN_facet = facet[np.int(result)]
    KNN_prob = knn.predict_proba(test)
    KNN_prob = np.round(KNN_prob,2)
    print (KNN_facet, KNN_prob)
    return [KNN_facet, KNN_prob]


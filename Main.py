import pandas as pd
import numpy as np

import Function
import NeuralNetwork
import TestYourSong

# Preprocessing----------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

#only running once to create spectograms and extract features from each song into data.csv file---------------
#path="./genres/"
#saveCSV ="data.csv"

#header=Function.Header()
#Function.OpeningCSV(header,saveCSV)
#Function.WritingToCSV(path,saveCSV)

#Analizing data.csv file---------------------------------------------------------------

data_all = pd.read_csv('data.csv')
data = data_all.drop(['filename'],axis=1)

#Encoding the Labels----------------------------------------------------------------------

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#Scaling the Feature columns--------------------------------------------------------------

scaler = StandardScaler()
x = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

#Dividing data into Training and Testing set----------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Classification with Keras--------------------------------

model=NeuralNetwork.BuildingNetwork(x_train, y_train, x_test, y_test)
NeuralNetwork.ValidatingApproach(x_train, y_train, x_test, y_test)

#Predictions on Test Data-----------------------------------------------------------------

predictions = model.predict(x_test)

#Testing on 10 random songs----------------------------------------------------------------------
count = 0
print("------------------------------ Prediction of random 10 song ------------------------------")
for x in range(10):
    print("Prediction: ", genres[np.argmax(predictions[x])],"\t Actual: ", genres[y_test[x]])
    if(genres[np.argmax(predictions[x])]==genres[y_test[x]]):
        count +=1

print("Pogodeno je: ", count, " od 10 pjesama!\n")
print("Ako zelite testirati svoje pjesme unesite D")

#Testing on new songs that we choose----------------------------------------------------------------------
inp=input()
if(inp=="D"):
    row_count = len(data.index)

    path="./TestYourSong/"
    saveCSV="./data.csv"

    Function.WritingToCSV(path,saveCSV)

    data = pd.read_csv('data.csv')
    data_name=data.iloc[:,0]
    data = data.drop(['filename'],axis=1)
    
    scaler = StandardScaler()
    x_yourSong = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

    print("\n------------------------------ Prediction of your song ------------------------------")
    predictions_yoursong = model.predict(x_yourSong)
    for x in range(row_count, len(x_yourSong)):
        print("Prediction of your song: ", genres[np.argmax(predictions_yoursong[x])],"\t Song name: ", data_name[x])

    data_all.to_csv('./data.csv',index=False)

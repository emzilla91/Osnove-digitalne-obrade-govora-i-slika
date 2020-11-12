import keras
from keras import models
from keras import layers

#Building Neural Network-----------------------------------------------------------------------------
def BuildingNetwork(x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=30,batch_size=128)
    test_loss, test_acc = model.evaluate(x_test,y_test)
    print('test accuracy: ',test_acc)
    print('test lost: ',test_loss)

    return model

#Validating our approach------------------------------------------------------------------

def ValidatingApproach(x_train, y_train, x_test, y_test):
    x_val = x_train[:200]
    partial_x_train = x_train[200:]

    y_val = y_train[:200]
    partial_y_train = y_train[200:]

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(partial_x_train,partial_y_train,epochs=50,batch_size=512,validation_data=(x_val, y_val))

    test_loss, test_acc = model.evaluate(x_test,y_test)

    print('test accuracy: ',test_acc)
    print('test lost: ',test_loss)

    return 0

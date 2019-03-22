from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
#change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
#part 3
#model.add(Dense(200, activation='tanh'))
#if you want to see the model trained with tanh, uncomment the previous line and switch the activation to tanh on the other two layers
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=0,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

#part 1
#just grab the results from history and plot them.
plt.plot(history.history['acc'],'b',label='training accuracy')
plt.plot(history.history['val_acc'],'r',label='validation accuracy')
plt.legend()
plt.title('Training Accuracy VS Validation Accuracy')
plt.show()

plt.plot(history.history['loss'],'b',label='training loss')
plt.plot(history.history['val_loss'],'r',label='validation loss')
plt.legend()
plt.title('Training Loss VS Validation Loss')
plt.show()

#looking at the results, I think it's safe to say that the model is not over-fitted. the loss and accuracy are pretty similar, all things considered
#and there aren't enough epochs to overfit any model, in my opinion.

#part 2
im = 10

pred = model.predict(train_data[im].reshape(1,784))
predval = np.argmax(pred)

#display the first image in the training data
plt.imshow(train_images[im,:,:],cmap='gray')
plt.title('Ground Truth : {0} Predicted {1}'.format(train_labels[im],predval))
plt.show()


#EC part 4

ims = [115,333,440,420] #must be size 4


for idx,im in enumerate(ims):
    p = model.predict(train_data[im].reshape(1,784))
    pv = np.argmax(p)
    plot = "22"+str(idx+1)
    #make a subplot for each image
    plt.subplot(plot,title="Truth: {0} Predicted: {1}".format(train_labels[im],pv)).imshow(train_images[im,:,:])

plt.show()
    
input()
input()

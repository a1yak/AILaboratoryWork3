import numpy as np
from numpy import unique, argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from cnnTensorModel import create_model

# loading dataset
(x_train, y_train), (x_test, y_test) = load_data()
# reshaping the training and testing data
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# normalizing the values of pixels of images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# determine the shape of the input images
img_shape = x_train.shape[1:]
print("Image Shape:", img_shape)

# defining the model
model = create_model(img_shape)

# model summary
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1)

# Visualizing Data Augmentation
image = x_train[421]
image = image.reshape((1,) + image.shape)  # Reshaping the image to (1, height, width, channels)

plt.figure(figsize=(10, 10))
plt.suptitle('Data Augmentation Example', fontsize=16)
i = 0
for batch in datagen.flow(image, batch_size=1):
    plt.subplot(3, 3, i+1)
    plt.grid(False)
    plt.imshow(batch.reshape(image.shape[1:3]), cmap='gray')
    if i == 8:
        break
    i += 1
plt.show()

# Calculate the number of steps per epoch for the given batch size
steps_per_epoch = len(x_train) // 128

# Callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=128, shuffle=True), 
                    epochs=40, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint, early_stopping],
                    verbose=2)  

# Load best model
model.load_weights("best_model.keras")

# Plotting accuracy and loss
plt.figure(figsize=(30, 10))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch NUMBER')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN loss')
plt.ylabel('Loss')
plt.xlabel('Epoch NUMBER')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()  

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Accuracy: {accuracy*100}%')

# Predict and show the first 15 images
images = x_test[:30]
predicted_labels = []

for i, image in enumerate(images):
    plt.subplot(3, 10, i + 1)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis('off')

    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    p = model.predict([image])
    predicted_label = np.argmax(p)
    predicted_labels.append(predicted_label)

    plt.title(f'Pred: {predicted_label}, Actual: {y_test[i]}', fontsize = 8)

plt.show()

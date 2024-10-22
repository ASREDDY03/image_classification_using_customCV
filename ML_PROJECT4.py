''' Import all the necessary libraries needed for runing the code keras is for creating a knn model
glob is for readng an image file '''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import tensorflow as tf

'''''Dataset paths for training and test data'''''
train_data_path = r"C:\Users\91868\Downloads\ML_DATASET\Monkey_Species_Data\Monkey_Species_Data\Training_Data"
test_data_path = r"C:\Users\91868\Downloads\ML_DATASET\Monkey_Species_Data\Monkey_Species_Data\Prediction_Data"

# Directory to save models
save_dir = r"C:\Users\91868\Documents\assignments\ABC\ML_ASSIGNMENT\ML_MODELS"
os.makedirs(save_dir, exist_ok=True)

''' Preprocessing: Remove corrupted JPEG files. Remove the files which are no JPEG format and only keep the desired images'''
dataset_path = r"C:\Users\91868\Downloads\ML_DATASET\Monkey_Species_Data\Monkey_Species_Data"
files = glob.glob(os.path.join(dataset_path, '**', '*'), recursive=True)
for file in files:
    if os.path.isfile(file):
        with open(file, "rb") as f:
            data = f.peek(10)  # Read the first 10 bytes for inspection
        if b"JFIF" not in data:
            print(f"Removing corrupt file: {file}")
            try:
                os.remove(file)
            except PermissionError as e:
                print(f"Error removing file {file}: {e}")

# Load the datasets

''' Load the datases and split them into training and test data and also define the image size so that it will not causes issues whicle training'''
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_path,
    label_mode="categorical", image_size=(100, 100), batch_size=32  
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_path,
    label_mode="categorical", image_size=(100, 100), shuffle=False, batch_size=32  
)

num_classes = len(train_data.class_names)

# Task 1: Check for saved models or create new ones

'''Checking the models is necessary as the cost of training is huge oe knn models'''
model1_path = os.path.join(save_dir, "model1.keras")
model2_path = os.path.join(save_dir, "model2.keras")

'''Model 1'''

'''if the models are not present in directory create the new ones'''
if os.path.exists(model1_path):
    model1 = load_model(model1_path)
else:
    model1 = models.Sequential([
        layers.Input(shape=(100, 100, 3)), 
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model1.fit(train_data, epochs=20, validation_data=test_data)
    model1.save(model1_path)


'''Model 2'''
'''if the models are not present in directory create the new ones'''
if os.path.exists(model2_path):
    model2 = load_model(model2_path)
else:
    model2 = models.Sequential([
        layers.Input(shape=(100, 100, 3)),  # Explicitly define input shape
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.fit(train_data, epochs=20, validation_data=test_data)
    model2.save(model2_path)

'''Evaluate Task 1 models to find the better model'''
test_images = []
test_labels = []
for images, labels in test_data:
    test_images.extend(images.numpy())
    test_labels.extend(np.argmax(labels.numpy(), axis=1))

test_images = np.array(test_images)
test_labels = np.array(test_labels)

predictions1 = model1.predict(test_images)
predictions2 = model2.predict(test_images)

pred_labels1 = np.argmax(predictions1, axis=1)
pred_labels2 = np.argmax(predictions2, axis=1)

accuracy1 = np.mean(pred_labels1 == test_labels)
accuracy2 = np.mean(pred_labels2 == test_labels)

'''Print the accuracies of the models to ensure the proper model is picked up and saved for later'''

print(f"Model 1 Accuracy: {accuracy1}")
print(f"Model 2 Accuracy: {accuracy2}")

better_model = model1 if accuracy1 > accuracy2 else model2
better_predictions = pred_labels1 if accuracy1 > accuracy2 else pred_labels2
better_model_name = "model1" if accuracy1 > accuracy2 else "model2"


'''Print the confusion matrix for the best model out of 2 models and print it'''

conf_matrix1 = confusion_matrix(test_labels, better_predictions)
print(f"Confusion Matrix for {better_model_name}:")
print(conf_matrix1)

# Task 2: Fine-tuning a Pre-Trained Model (EfficientNet)

'''use the pretrained model for improving the accuracy of the best model and tune it untill convergence but we are limiting the code to 20 epoch as the training time is huge and more over there is convergence within the mdoel at that time'''

model3_path = os.path.join(save_dir, "model3_fine_tuned.keras")
if os.path.exists(model3_path):
    model3 = load_model(model3_path)
else:
    base_model = tf.keras.applications.EfficientNetV2S(include_top=False, input_shape=(100, 100, 3))
    base_model.trainable = False
    model3 = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model3.fit(train_data, epochs=20, validation_data=test_data)
    model3.save(model3_path)
test_loss3, test_acc3 = model3.evaluate(test_data)


'''Print the accuracy and confusion matrix of the FIne_tuned model'''
print(f"Fine-tuned Model Accuracy: {test_acc3}")

# Confusion Matrix for the fine-tuned model
fine_tuned_preds = model3.predict(test_images)
fine_tuned_labels = np.argmax(fine_tuned_preds, axis=1)
conf_matrix2 = confusion_matrix(test_labels, fine_tuned_labels)
print("Confusion Matrix for Fine-tuned Model:")
print(conf_matrix2)

# Task 3: Error Analysis
# Select 10 incorrect predictions from the better Task 1 model
incorrect_indices = np.where(better_predictions != test_labels)[0][:10]

# Analyze the predictions and visualize
for i, idx in enumerate(incorrect_indices):
    img = test_images[idx]
    actual_label = test_data.class_names[test_labels[idx]]
    task1_pred_label = test_data.class_names[better_predictions[idx]]
    task2_pred_label = test_data.class_names[fine_tuned_labels[idx]]

    print(f"Image {i+1}")
    print(f"  Actual: {actual_label}")
    print(f"  Predicted by Task 1 ({better_model_name}): {task1_pred_label}")
    print(f"  Predicted by Task 2 (fine-tuned model): {task2_pred_label}")

    plt.imshow(img.astype("uint8"))
    plt.title(f"Actual: {actual_label}, Task 1 Pred: {task1_pred_label}, Task 2 Pred: {task2_pred_label}")
    plt.show()

    # Check if Task 2 improved
    if task1_pred_label != actual_label and task2_pred_label == actual_label:
        print(f"Task 2 model corrected the prediction for image {i+1}")
    elif task1_pred_label != actual_label and task2_pred_label != actual_label:
        print(f"Task 2 model did not correct the prediction for image {i+1}")
    else:
        print(f"Task 2 model made the same correct prediction for image {i+1}")

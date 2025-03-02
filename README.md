#Monkey Species Classification using CNN

Project Overview

This project implements a Convolutional Neural Network (CNN) model to classify monkey species using image data. The models are trained using custom-built CNNs and transfer learning with EfficientNet to improve classification accuracy. The pipeline also includes data preprocessing, automated model selection, and fine-tuning for optimal performance.

Dataset

The dataset consists of monkey species images categorized into different classes. The training and test datasets are structured in directories:

Training Data Path: ML_DATASET/Monkey_Species_Data/Training_Data

Test Data Path: ML_DATASET/Monkey_Species_Data/Prediction_Data

Model Architecture

Custom CNN Models

Two CNN architectures were developed and evaluated:

Model 1:

Conv2D (32 filters, 3x3, ReLU) → MaxPooling (2x2)

Conv2D (64 filters, 3x3, ReLU) → MaxPooling (2x2)

Flatten → Dense (128 neurons, ReLU) → Dropout (0.5)

Output Layer: Dense (softmax activation for multi-class classification)

Model 2 (Deeper architecture):

Conv2D (64 filters, 3x3, ReLU) → MaxPooling (2x2)

Conv2D (128 filters, 3x3, ReLU) → MaxPooling (2x2)

Conv2D (256 filters, 3x3, ReLU) → MaxPooling (2x2)

Flatten → Dense (512 neurons, ReLU) → Dropout (0.5)

Output Layer: Dense (softmax activation)

Pre-Trained Model (EfficientNetV2S - Fine-Tuned)

A pre-trained EfficientNetV2S model was fine-tuned to enhance accuracy. The model was trained with frozen base layers and fine-tuned for 20 epochs.

Preprocessing Steps

Removal of Corrupt Images: Images without proper encoding were filtered out.

Image Augmentation: Applied rotation, flipping, and zooming to enhance generalization.

Resizing: Standardized all images to 100x100 pixels.

One-Hot Encoding: Converted categorical labels to numeric form.

Training & Evaluation

Training Configuration

Optimizer: Adam (Learning Rate: 0.0001)

Loss Function: Categorical Crossentropy

Batch Size: 32

Epochs: 20

Model Evaluation

Accuracy Metrics:

Model 1: X%

Model 2: Y%

Fine-Tuned Model: Z%

Confusion Matrices were used to analyze misclassifications.

Deployment

Model Selection: The best model is saved and loaded for inference.

Saved Model Directory: ML_ASSIGNMENT/ML_MODELS

Formats: .keras models are stored for future use.

Error Analysis

Incorrect Predictions: The top 10 misclassified images were analyzed.

Comparison of Model Predictions: Task 1 (Custom CNN) vs. Task 2 (Fine-Tuned Model).

Visualization: Matplotlib was used to display misclassified images and corrections.

Future Enhancements

Hyperparameter Tuning: Grid search to optimize dropout rates and learning rates.

Edge Deployment: Convert the best-performing model to TFLite for mobile inference.

Explainability: Implement Grad-CAM to interpret CNN feature maps.

How to Run the Code

Install Dependencies

pip install tensorflow numpy matplotlib

Run the Training Script

python train_monkey_species.py

Evaluate the Model

python evaluate_model.py



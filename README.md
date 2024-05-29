# Imagined-Emotions-Classification-with-EEG-Data

## Introduction

This project aims to classify imagined emotions using EEG (Electroencephalogram) data. EEG is a method to record electrical activity of the brain. The specific emotions considered in this project are Angry, Disgust, Fear, Happy, Sad, and Surprise. A Convolutional Neural Network (CNN) is employed to perform this classification task.

## Workflow

### 1. Data Preparation

**Mounting Google Drive:**
- The data is stored on Google Drive and needs to be accessed through Google Colab. The drive is mounted to access the data files.

**Setting Directory Path:**
- The directory path where the EEG data files are stored is defined. This path is used to read the data for further processing.

### 2. Loading Data

**Loading EEG Signals:**
- The project includes data for six emotions. Each emotion's data is stored in separate directories.
- Each signal file is loaded, and the data is extracted and transposed for consistency.
- Labels are assigned to each signal based on the emotion it represents.

### 3. Data Wrangling

**Creating DataFrame:**
- A pandas DataFrame is created to organize the EEG features and corresponding labels.

**Conversion to Numpy Arrays:**
- The features and labels are converted into numpy arrays for easier manipulation and compatibility with machine learning models.

### 4. Splitting Data

**Training and Testing Sets:**
- The data is split into training and testing sets using a 75-25 split ratio. This allows for training the model on one subset of data and validating its performance on another.

### 5. Model Building

**Defining the Model Architecture:**
- A Convolutional Neural Network (CNN) is chosen due to its effectiveness in handling spatial data, such as EEG signals.
- The model consists of several Conv1D layers with Batch Normalization and Dropout layers. These layers help in feature extraction and regularization.

**Layers Explanation:**
- **Input Layer:** The input shape is specified to match the dimensions of the EEG data.
- **Conv1D Layers:** These layers apply convolution operations to extract features from the input data. Multiple convolution layers with varying filters and kernel sizes are used.
- **Batch Normalization:** This layer normalizes the output of the previous layer, improving training stability.
- **Dropout Layers:** Dropout is used to prevent overfitting by randomly setting a fraction of input units to zero during training.
- **Flatten Layer:** Converts the multi-dimensional output of the convolutional layers into a single dimension.
- **Dense Layers:** Fully connected layers that further process the extracted features. The final dense layer uses a softmax activation function to output probabilities for each of the six emotions.

### 6. Model Training

**Compiling the Model:**
- The model is compiled with the loss function `sparse_categorical_crossentropy` and the optimizer `adam`.

**Fitting the Model:**
- The model is trained on the training data for a specified number of epochs. During training, the model's performance is also evaluated on the validation set.

### 7. Model Evaluation

**Evaluating Model Performance:**
- After training, the model's performance is evaluated on the test set. Metrics such as accuracy and loss are reported.

### 8. Visualization

**Plotting Accuracy and Loss:**
- Training and validation accuracy and loss are plotted to visualize the model's performance over epochs. This helps in understanding the learning behavior and potential overfitting or underfitting.

### 9. Predictions

**Making Predictions:**
- The trained model is used to make predictions on the test set. The predicted labels are compared with the true labels to evaluate the model's accuracy.

## Conclusion

This project successfully demonstrates the process of classifying imagined emotions using EEG data and a Convolutional Neural Network. The model's architecture and the preprocessing steps are critical to achieving good performance. Visualization of training metrics provides insights into the model's learning process, helping to fine-tune and improve the model further.


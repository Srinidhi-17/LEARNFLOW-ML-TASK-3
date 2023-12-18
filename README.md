Project Title:
Speech Command Recognition using Convolutional Neural Networks (CNN)

Project Description:
This project focuses on training a CNN model for recognizing speech commands. It utilizes the Google Speech Commands dataset, preprocesses the audio data, and trains a CNN model for classification.

Dataset Information:
The dataset used for this project is the Google Speech Commands dataset, which contains a collection of short audio commands spoken by different users.https://kaggle.com/neehakurelli/google-speech-commands


Requirements:
Python 3.x
NumPy
librosa
scikit-learn
TensorFlow
Keras

Installation Instructions:
Install dependencies: pip install -r requirements.txt
Download the Google Speech Commands dataset. 
Set up the project environment.

Usage Instructions:
Open and run the Jupyter notebook or Python script containing the provided code.
Monitor the training progress with accuracy information.
Evaluate the trained model on the testing dataset.
Visualize the confusion matrix and classification report.
Predict the class for a sample audio command.

Model Architecture:
The model architecture consists of a Conv2D layer, MaxPooling2D layer, Flatten layer, Dense layer with ReLU activation, Dropout layer, and a Dense output layer with softmax activation.

Results:
The trained model achieved an accuracy of 0.08 on the testing dataset.

Evaluation Metrics:
The project uses the sparse categorical crossentropy loss function and accuracy as the evaluation metric.

Preprocessing Details:
Audio data is preprocessed using librosa to extract Mel-frequency cepstral coefficients (MFCCs). The data is then formatted and padded to meet the input requirements of the CNN model.

Visuals:
The project includes visualizations such as accuracy plots during training, a confusion matrix, and a classification report. Find sample audio files and their predictions in the visuals directory.

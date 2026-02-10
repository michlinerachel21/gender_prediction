Gender Prediction Using CNN (TensorFlow & Keras)
📌 Project Overview
This project implements a Convolutional Neural Network (CNN) to predict gender (Male/Female) from facial images using TensorFlow and Keras.
The model is trained on labeled image data organized into Training and Validation directories and can also predict gender for a single input image.

🧠 Model Architecture
The CNN model consists of:
3 Convolutional layers with ReLU activation
Max Pooling layers for feature reduction
Fully connected Dense layer
Dropout layer to prevent overfitting
Sigmoid output layer for binary classification

📂 Dataset Structure
The dataset should be organized as follows:
gender_pred/
│
├── Training/
│   ├── male/
│   └── female/
│
├── Validation/
│   ├── male/
│   └── female/
Each folder contains face images corresponding to the class.

⚙️ Technologies Used
Python
TensorFlow / Keras
OpenCV (optional for image handling)
NumPy

🔄 Data Preprocessing
Images resized to 128 × 128
Pixel values normalized using rescale = 1./255

Data augmentation applied on training data:
Rotation
Zoom
Horizontal flip

🚀 Model Training
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy
Epochs: 10
Batch Size: 32
Training is performed using ImageDataGenerator with real-time augmentation.

📈 Model Evaluation
Validation data is used during training to monitor:
Validation Accuracy
Validation Loss

Class labels are verified using:
print(train_data.class_indices)


🧪 Prediction on New Image
The trained model predicts gender for a single image:
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Male")
else:
    print("Female")


Output > 0.5 → Male
Output ≤ 0.5 → Female

💾 Model Saving
The trained model is saved for reuse:
model.save("gender_prediction_cnn.h5")


📌 How to Run the Project
Install required libraries:
pip install tensorflow numpy opencv-python
Prepare dataset in the required folder structure
Update dataset and image paths in the code
Run the Python script
Use the saved model for prediction

📊 Output
Trained CNN model
Gender prediction for test images
Training and validation accuracy logs


✅ Use Cases
Face-based gender classification
Learning CNN image classification
Academic and mini-project implementation




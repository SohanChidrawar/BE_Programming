import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = load_model("letter_recognition_model.h5")
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_encoder_classes.npy",allow_pickle=True)

# Function to preprocess input dimensions
def preprocess_input(dimensions):
    # Convert dimensions to a numpy array and reshape it to match the input shape of the model
    input_data = np.array(dimensions).reshape(1, -1)
    return input_data

# Function to predict the output using the trained model
def predict_output(dimensions):
    # Preprocess the input dimensions
    input_data = preprocess_input(dimensions)
    # Use the model to predict the output
    predicted_class = np.argmax(model.predict(input_data), axis=-1)
    # Convert the predicted class index back to the original letter
    predicted_letter = label_encoder.inverse_transform(predicted_class)
    return predicted_letter[0]

# Function to handle button click event
def predict():
    # Get input dimensions from the user
    dimensions_str = dimension_entry.get()
    dimensions = list(map(int, dimensions_str.split(',')))
    # Predict the output
    predicted_letter = predict_output(dimensions)
    # Show the predicted letter in a message box
    messagebox.showinfo("Prediction Result", f"Predicted Letter: {predicted_letter}")

# Create the main window
root = tk.Tk()
root.title("Text Prediction")

# Create label and entry for input dimensions
dimension_label = tk.Label(root, text="Enter dimensions separated by commas:")
dimension_label.pack()

dimension_entry = tk.Entry(root)
dimension_entry.pack()

# Create button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Run the main event loop
root.mainloop()


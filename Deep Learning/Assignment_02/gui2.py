import tkinter as tk
from tkinter import ttk
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np

#load tokenizer
tokenizer = Tokenizer(num_words=10000)

#load trained model
model = load_model('sentiment_model.h5')

#define lablels
labels= [ 'Negative' , 'Positive']

def classify_review(review_text):
    # Tokenize and pad sequence
    sequence = tokenizer.texts_to_sequences([review_text])
    sequence_padded = pad_sequences(sequence, maxlen=200)
    # Make prediction using the trained model
    prediction = model.predict(sequence_padded)
    # Convert prediction to label
    label = labels[np.argmax(prediction)]
    return label

def classify_and_display():
    review_text = review_entry.get("1.0", "end-1c")
    prediction = classify_review(review_text)
    result_label.config(text="Predicted Sentiment: {prediction}")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis")

# Create text entry for input review
review_label = ttk.Label(root, text="Enter your review:")
review_label.grid(row=0, column=0, padx=5, pady=5)

review_entry = tk.Text(root, height=5, width=50)
review_entry.grid(row=0, column=1, padx=5, pady=5)

# Create button to classify review
classify_button = ttk.Button(root, text="Classify", command=classify_and_display)
classify_button.grid(row=1, columnspan=2, padx=5, pady=5)

# Create label to display result
result_label = ttk.Label(root, text="")
result_label.grid(row=2, columnspan=2, padx=5, pady=5)

# Run the GUI event loop
root.mainloop()

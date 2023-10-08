

#pip install tensorflow
#pip install numpy
#pip install cv
#pip install opencv-python
#pip install tk
#pip install shutil
#pip install pillow

import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter.messagebox import showinfo
import shutil
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2

def openDermDetectGUI():
    global filename, canvas, middle_frame, submit_button, skin_disease_label

    # TkInter
    root = tk.Tk()
    root.title("DermDetect")
    root.geometry("400x400")

    # Set background color to #FAF9F6
    root.configure(bg="#FAF9F6")

    title = ttk.Label(root, text="DermDetect", font=("Arial", 20))
    title.pack(pady=10)

    # Middle Frame
    middle_frame = tk.Frame(root, bg="#FAF9F6")
    middle_frame.pack(expand=True, fill='both')

    # Canvas for displaying the uploaded image
    canvas = tk.Canvas(middle_frame, width=200, height=200, bg="white", highlightthickness=0)
    canvas.pack(pady=20)

    # Label for skin disease prediction
    skin_disease_label = ttk.Label(middle_frame, text="No image uploaded", font=("Arial", 12))
    skin_disease_label.pack(pady=20)

    # File chooser
    filename = None

    def chooseFile():
        global filename

        fileTypes = (
            ('JPEG files', '*.jpg;*.jpeg'),
            ('PNG files', '*.png'),
        )

        filename = fd.askopenfilename(
            title='Open an image',
            initialdir='/',
            filetypes=fileTypes)

        showinfo(
            title='Selected Image',
            message=filename
        )

        display_image()

    def display_image():
        global submit_button

        if filename:
            img = Image.open(filename)
            img.thumbnail((200, 200))
            img = ImageTk.PhotoImage(img)

            canvas.create_image(0, 0, anchor=tk.NW, image=img)

            canvas.image = img
            submit_button.config(state=tk.NORMAL)  # Enable the submit button

    upload_button = ttk.Button(root, text="Upload Image", command=chooseFile)
    upload_button.pack(side=tk.BOTTOM, pady=20)

    submit_button = ttk.Button(root, text="Submit", command=submit)
    submit_button.pack(side=tk.BOTTOM, pady=10)
    submit_button.config(state=tk.DISABLED)  # Disable the submit button initially

def predict_skin_disease(image_path):
    class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea", "Vitiligo"]

    # Load saved model
    model = tf.keras.models.load_model('/content/drive/My Drive/kaggle-skin/6claass.h5')

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = vgg_model.predict(img)
    img = img.reshape(1, -1)

    # Make a prediction on the preprocessed image
    print(class_names)
    pred = model.predict(img)[0]

    pred[0] = pred[0] * 3
    print(pred)

    predicted_class_index = np.argmax(pred)
    predicted_class_second = np.argsort(pred)[-2]

    predicted_class_name = class_names[predicted_class_index]
    predicted_second_class_name = class_names[predicted_class_second]

    return predicted_class_name, predicted_second_class_name

def submit():
    global filename, middle_frame, skin_disease_label

    if not filename:
        showinfo("No Image Selected", "Please upload an image.")
        return

    # Save the uploaded image in a directory
    image_path = f"./images/{filename.split('/')[-1]}"
    shutil.copyfile(filename, image_path)

    # Perform skin disease prediction
    predicted_disease, predicted_second_disease = predict_skin_disease(image_path)
    
    # Update the skin disease label
    skin_disease_label.config(text=f"Predicted Skin Disease: {predicted_disease}")
    showinfo("Image Submitted", f"Image successfully submitted. Predicted Skin Disease: {predicted_disease}")

    # After submission, center the image and disable the submit button
    skin_disease_label.pack_forget()
    skin_disease_label.pack(in_=middle_frame, pady=20)
    submit_button.config(state=tk.DISABLED)

if __name__ == '__main__':
    # Load VGG model
    vgg_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

    openDermDetectGUI()
    tk.mainloop()

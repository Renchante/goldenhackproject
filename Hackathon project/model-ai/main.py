

#pip install tensorflow
#pip install numpy
#pip install cv
#pip install opencv-python
#pip install tk
#pip install shutil
#pip install pillow



#import tensorflow as tf
import numpy as np
import cv2
#from tensorflow.keras.applications import  VGG19
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter.messagebox import showinfo
import shutil
from PIL import Image, ImageTk


def openDermDetectGUI():
    global filename, image_label, middle_frame

    # TkInter
    root = tk.Tk()
    root.title("DermDetect")
    root.geometry("1000x650")

    # Set background color to #FAF9F6
    root.configure(bg="#FAF9F6")

    title = ttk.Label(root, text="DermDetect", font=("Arial", 20))
    title.pack(pady=10)

    # Middle Frame
    middle_frame = tk.Frame(root, bg="#FAF9F6")
    middle_frame.pack(expand=True, fill='both')

    # Label for displaying the uploaded image
    image_label = ttk.Label(middle_frame, text="No image uploaded", font=("Arial", 12))
    image_label.pack(pady=20)

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
        if filename:
            img = Image.open(filename)
            img.thumbnail((700, 700))
            img = ImageTk.PhotoImage(img)

            image_label.config(image=img)
            image_label.image = img

    upload_button = ttk.Button(root, text="Upload Image", command=chooseFile)
    upload_button.pack(side=tk.BOTTOM, pady=20)

    submit_button = ttk.Button(root, text="Submit", command=submit)
    submit_button.pack(side=tk.BOTTOM, pady=10)
    

def submit():
    global filename, middle_frame

    if not filename:
        showinfo("No Image Selected", "Please upload an image.")
        return

    # Save the uploaded image in a directory
    image_path = f"./images/{filename.split('/')[-1]}"

    shutil.copyfile(filename, image_path)
    print(filename,image_path)

    showinfo("Image Submitted", "Image successfully submitted!")
    

    # After submission, center the image
    image_label.pack_forget()
    image_label.pack(in_=middle_frame, pady=20)

if __name__ == '__main__':
    openDermDetectGUI()
    tk.mainloop()









i=1
# tests for normal
if i ==0:
    vgg_model = VGG19(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3)) 

    def predict_skin_disease(image_path):
        class_names = ["Acne","Eczema","Atopic","Psoriasis","Tinea","vitiligo"]

        # Load saved model
        model = tf.keras.models.load_model('/content/drive/My Drive/kaggle-skin/6claass.h5')

        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (180, 180))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        img = vgg_model.predict(img)
        img = img.reshape(1, -1)

        # Make prediction on preprocessed image
        print(class_names)
        pred = model.predict(img)[0]
        
        pred[0]=pred[0]*3
        print(pred)

        predicted_class_index = np.argmax(pred)
        predicted_class_second = np.argsort(pred)[-2]
    
        predicted_class_name = class_names[predicted_class_index]
        predicted_second_class_name = class_names[predicted_class_second]

        return predicted_class_name, predicted_second_class_name

    #print(predict_skin_disease("/content/drive/My Drive/kaggle-skin/skindatasets/test/Normal/0_0_aidai_0029.jpg")) #("/kaggle/input/skindat
    #"/kaggle/input/skindatasets/skin/test/Normal/0_0_aidai_0029.jpg"

    print("Eczema")
    print(predict_skin_disease("/content/drive/My Drive/kaggle-skin/realtest/eczema-example.jpg"))
    print()

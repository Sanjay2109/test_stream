import streamlit as st
from PIL import Image
import tensorflow
import numpy as np
import pickle

classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Vehicle > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing vehicle > 3.5 tons'}

# def load_model(model_path):
#     """
#   Loads a saved model from a specified path.
#   """
#     print(f"Loading saved model from: {model_path}")
#     real_one = tensorflow.keras.models.load_model(model_path)
#     print("Finished loading the model.\n")
#     return real_one


# model = load_model("traffic-sign-model")
pickled_model = pickle.load(open("pkl_model.pkl", "rb"))

def predict_pkl(img):
    image = Image.open(img)
    image = image.resize((30, 30))
    data = [np.array(image)]
    X_test = np.array(data)
    return pickled_model.predict(X_test)


def predict(img):
    image = Image.open(img)
    image = image.resize((30, 30))
    data = [np.array(image)]
    X_test = np.array(data)
    return model.predict(X_test)


def load_image(image):
    return Image.open(image)


image_file = st.file_uploader("Choose an Image", type=["JPG ", "PNG", "JPEG", "TIFF"])
st.write("Note that the predicted classes are from here:")
st.write(classes)
st.write("Submit pictured containing only the sign board, or else it may predict incorrect classes.")

if image_file is not None:
    file_details = {"file_name": image_file.name, "file_type": image_file.type,
                    "file_size": image_file.size}
    st.write(f"File name: {file_details['file_name']}")
    st.write(f"File size: {file_details['file_size']}")
    st.write(f"File type: {file_details['file_type']}")
    st.image(load_image(image_file), width=250)

    predictions = predict_pkl(image_file)
    listed_predictions = predictions.flatten().tolist()
    max_prediction_index = np.argmax(listed_predictions)
    st.write(classes[max_prediction_index])

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

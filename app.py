from turtle import up
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions
from PIL import Image
import pandas as pd
import requests
from bs4 import BeautifulSoup

st.title("Image Recognition App")

# Load pre-trained model (you can also use your own model)

model = tf.keras.applications.ResNet50(weights="imagenet")

# Ask user to choose between uploading an image or providing a URL
option = st.radio(
    "Select an option:", ("Upload an image", "Provide the URL of an image")
)

if option == "Upload an image":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    # Ask user to provide the URL of the image
    url = st.text_input("Enter the URL of the image you want to classify")
    if url:
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            st.image(image, caption="Image from URL", use_column_width=True)
        except:
            st.write("Invalid URL or unable to access the image. Please try again.")

if "image" in locals():
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    items = decode_predictions(predictions, top=3)

    def create_df_from_items(items):
        names = []
        confidence = []
        for item in items[0]:
            names.append(item[1])
            confidence.append(item[2])
        df = pd.DataFrame({"names": names, "confidence": confidence})
        # format the confidence column to be a percentage wiyh 2 decimal places
        df["confidence"] = df["confidence"].apply(lambda x: x * 100)
        df["confidence"] = df["confidence"].apply(lambda x: round(x, 2))
        # add the % sign
        df["confidence"] = df["confidence"].apply(lambda x: str(x) + "%")

        return df

    list_of_items = create_df_from_items(items)["names"].tolist()
    # Display the results
    st.write(f"This image most likely contains the following items:")
    st.dataframe(create_df_from_items(items))

    search_results = []
    # search the web and retetrieve some cool facts about the items
    for item in list_of_items:
        # search for information about "banana" on Google
        query = item
        url = f"https://www.google.com/search?q={query}"
        response = requests.get(url)

        # parse the HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # extract the search results
        search_results = soup.find_all("div", class_="g")
        st.write(f"{search_results}")
        # print the first search result
        search_results.append(search_results[0].text)

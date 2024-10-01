import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
import pickle

# Download stopwords for NLTK
nltk.download('stopwords')

# Load the saved Keras model
model = load_model('my_model.h5')

# Load the saved CountVectorizer
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# Genre assignment dictionary
assign = {'Horror': 1, 'Documentary': 2, 'New': 3, 'Classics': 4, 'Games': 5, 'Sci-Fi': 6,
          'Foreign': 7, 'Family': 8, 'Travel': 9, 'Music': 10, 'Sports': 11, 'Comedy': 12,
          'Drama': 13, 'Action': 14, 'Children': 15, 'Animation': 16}

# Reverse the genre dictionary to map numeric labels to genre names
genre_mapping = {v: k for k, v in assign.items()}

def decode_prediction(prediction):
    # Get the index of the highest probability in the prediction array
    predicted_index = prediction.argmax()

    # Map the predicted index to a genre name using the reversed genre_mapping
    predicted_genre = genre_mapping.get(predicted_index, "Unknown Genre")

    return predicted_genre

# Streamlit app input
st.title('Movie Genre Prediction')
input_text = st.text_area("Enter a movie description:")

if st.button("Predict"):
    if input_text:
        # Preprocess input text
        text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=input_text)
        text = text.lower()
        words = text.split()
        clean_words = [word for word in words if word not in stop_words]
        stemmed_words = [ps.stem(word) for word in clean_words]
        preprocessed_text = " ".join(stemmed_words)

        # Vectorize the preprocessed text
        corpus = [preprocessed_text]
        x = cv.transform(corpus).toarray()

        # Make the prediction using the loaded model
        prediction = model.predict(x)

        # Decode the prediction to a genre
        predicted_genre = decode_prediction(prediction)
        st.write(f"Predicted genre: {predicted_genre}")

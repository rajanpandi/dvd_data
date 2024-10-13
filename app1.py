import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler  # Assuming this is used to standardize input data

# Load the pre-trained model
def load_model():
    model_path = '/home/ubuntu/dvd_data/my_model.hdf5'  # Ensure this path is correct
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    return model

# Instantiate scaler if needed (fit it based on your training data)
sc = StandardScaler()

model = load_model()

def predict(input_data):
    # Reshape the input data for the model
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    input_data_standardized = (input_data_reshaped)  # Apply standardization if needed

    # Make prediction
    prediction = model.predict(input_data_standardized)

    # Movie categories mapping
    movie = {
        'Horror': 1, 'Documentary': 2, 'New': 3, 'Classics': 4, 'Games': 5,
        'Sci-Fi': 6, 'Foreign': 7, 'Family': 8, 'Travel': 9, 'Music': 10,
        'Sports': 11, 'Comedy': 12, 'Drama': 13, 'Action': 14, 'Children': 15,
        'Animation': 16
    }

    # Process predictions to get the predicted category
    max_index = np.argmax(prediction[0])  # Get the index of the predicted category
    predicted_category = list(movie.keys())[max_index]

    return predicted_category

def main():
    st.title("Movie Category Prediction")

    # Input fields for user
    rating = st.selectbox("Rating", options=[0, 1, 2, 3, 4])  # Dropdown for ratings

    # Define customer ID and actor ID ranges as per your request
    customer_id_options = np.array([340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352,
       353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365,
       366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378,
       379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
       392, 393, 394, 395, 396, 397, 398, 399, 401, 402, 403, 404, 405,
       406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
       419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
       433, 434, 435, 436, 437, 438, 440, 441, 442, 443, 445, 446, 447,
       448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460,
       461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473,
       475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487,
       488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500,
       501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513,
       514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526,
       527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,
       540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552,
       553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565,
       566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578,
       579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
       592, 593, 594, 595, 596, 597, 598])
    customer_id = st.selectbox("Customer ID", options=customer_id_options)  # Dropdown for customer ID

    actor_id_options = np.array([160, 126,  76,  58,   0, 185, 135,  77,  49,   8, 190, 149,  94,
        93,  81,  59,  57,  27, 106,  92,  74,  61,  19,  14, 176, 146,
        99, 129, 127,  70, 121, 113, 109,  86,  52, 150, 144,  80,  41,
       123, 120,  28,  22, 181, 171, 137, 105, 102,  60, 177, 162, 180])
    actor_id = st.selectbox("Actor ID", options=actor_id_options)  # Dropdown for actor ID

    # Additional fields
    rental_duration = st.selectbox("rental_duration", options=[0, 1, 2, 3, 4])
    rental_rate = st.selectbox("rental_rate", options=[0, 1, 2])
    length_options = np.array([107,  84,  92,  54, 119,  40, 115,  39,  73,  46,  36,  68, 132,
       134,  86,   1, 110,  12,  76, 125,  98, 129,  53,  28,  41, 101,
        74,   7,  66, 104,  19,  57,   6,  32,  89,  97,  55,  75,  56,
       139,  60, 136,  18, 102, 108,  79,  58,  77,  82,  34, 128,  17,
        27,  85,  65,  96,  29,  30,  51,  38, 117,  15, 122, 127,  70,
       106,   8,  80, 116, 121, 126,  91,  63,   0,  33,  67,  52,  93,
       100, 120,  16, 130,   9,  44,  62,  31,  22,  14,   5, 138,  42,
       114,  43,  49,  78,  90,  37,  64,   2,  47,  99,  26,  83, 137,
        71, 109,  35, 123,  87,  72,  11, 133, 113, 105,  21,  88, 112,
       118,   4,  13, 111,   3,  24,  95,  45,  59,  61, 135,  94,  50,
        23,  48,  10,  81, 131, 103, 124,  69,  25,  20])
    length = st.selectbox("length", options=length_options)  # Dropdown for actor ID
    # length = st.number_input("Length")
    inventory_id = st.number_input("Inventory ID", step=1, format="%d")
    store_id = st.selectbox("store_id", options=[0, 1])

    if st.button("Predict"):
        # Prepare input data
        input_data = [rating, customer_id, actor_id, rental_duration, rental_rate, length, inventory_id, store_id]
        
        # Call prediction function
        result = predict(input_data)
        
        # Display the predicted movie category
        st.success(f"Predicted Movie Category: {result}")

if __name__ == "__main__":
    main()

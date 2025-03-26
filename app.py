
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


model = load_model("movie_recommender.h5")

# Load datasets
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# Encode userId and movieId
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["user"] = user_encoder.fit_transform(ratings["userId"])
ratings["movie"] = movie_encoder.fit_transform(ratings["movieId"])

num_users = ratings["user"].max() + 1
num_movies = ratings["movie"].max() + 1

# Streamlit App UI
st.title("🎬 Movie Recommender System")


user_id = st.selectbox("Select a User ID:", ratings["userId"].unique())

# Predict function
def recommend_movies(user_id, top_n=10):
    if user_id not in ratings["userId"].values:
        st.error("User ID not found in dataset.")
        return pd.DataFrame()
    
    user_index = user_encoder.transform([user_id])[0]
    movie_indices = np.arange(num_movies)

    predictions = model.predict([np.array([user_index] * num_movies), movie_indices])
    
    top_movie_indices = np.argsort(predictions.flatten())[-top_n:][::-1]
    recommended_movie_ids = movie_encoder.inverse_transform(top_movie_indices)
    
    recommended_movies = movies[movies["movieId"].isin(recommended_movie_ids)]
    return recommended_movies[["movieId", "title"]]

# Display recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies(user_id)
    if not recommendations.empty:
        st.write("### 🎥 Recommended Movies for User", user_id)
        st.table(recommendations)
    else:
        st.error("No recommendations available.")



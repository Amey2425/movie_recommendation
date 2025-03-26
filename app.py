# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import pandas as pd

# # Load the trained recommendation model
# model = tf.keras.models.load_model("movie_recommender.h5")

# # Load movies dataset
# movies_df = pd.read_csv("ml-latest-small/movies.csv")  # Ensure the file exists

# # Map movie IDs to sequential indices
# unique_movie_ids = sorted(movies_df["movieId"].unique())  # Sorted for consistency
# movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
# index_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_index.items()}  # Reverse mapping

# # Streamlit UI
# st.title("🎬 Movie Recommendation System")

# # User input for recommendation
# user_id = st.number_input("Enter User ID:", min_value=1, step=1)
# top_n = st.slider("Number of Recommendations:", 1, 10, 5)

# if st.button("Get Recommendations"):
#     # Convert movie IDs to indices
#     valid_movie_ids = np.array([movie_id_to_index[movieId] for movieId in movies_df["movieId"]])

#     # Ensure user_ids is correctly formatted
#     user_ids = np.full(len(valid_movie_ids), user_id)

#     # Predict ratings for all movies for the given user
#     print(user_ids.shape, valid_movie_ids.shape)
#     user_ids = user_ids - 1
#     valid_movie_ids = valid_movie_ids - 1


#     predictions = model.predict([user_ids, valid_movie_ids]).flatten()

#     # Get top N recommended movie indices
#     top_indices = predictions.argsort()[-top_n:][::-1]
#     recommended_movie_ids = [index_to_movie_id[idx] for idx in top_indices]  # Convert back to original IDs

#     # Get movie details
#     recommended_movies = movies_df[movies_df["movieId"].isin(recommended_movie_ids)]

#     # Display results
#     st.subheader(f"Top {top_n} Recommended Movies for User {user_id}:")
#     for _, row in recommended_movies.iterrows():
#         st.write(f"🎥 **{row['title']}** (Movie ID: {row['movieId']})")
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
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

# Select user ID
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



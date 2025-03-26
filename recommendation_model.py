import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Dense, Concatenate, Dropout, Add, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load dataset
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

# Encode userId and movieId
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["user"] = user_encoder.fit_transform(ratings["userId"])
ratings["movie"] = movie_encoder.fit_transform(ratings["movieId"])

num_users = ratings["user"].max() + 1  # Fix: Ensure correct input_dim
num_movies = ratings["movie"].max() + 1  # Fix: Account for max index, not just unique count

# Train-test split
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

X_train = [train["user"].values, train["movie"].values]
y_train = train["rating"].values

X_test = [test["user"].values, test["movie"].values]
y_test = test["rating"].values

# Define optimized model
embedding_dim = 50  # Adjusted for generalization

def create_model():
    user_input = Input(shape=(1,))
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, embeddings_regularizer=l2(0.001))(user_input)
    user_bias = Embedding(input_dim=num_users, output_dim=1)(user_input)
    user_vec = Flatten()(user_embedding)
    user_bias_vec = Flatten()(user_bias)
    
    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim, embeddings_regularizer=l2(0.001))(movie_input)
    movie_bias = Embedding(input_dim=num_movies, output_dim=1)(movie_input)
    movie_vec = Flatten()(movie_embedding)
    movie_bias_vec = Flatten()(movie_bias)
    
    # Concatenate embeddings
    concat = Concatenate()([user_vec, movie_vec])
    
    # Fully connected layers with Batch Normalization & Dropout
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.002))(concat)
    bn1 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.4)(bn1)

    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.002))(dropout1)
    bn2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.4)(bn2)

    dense3 = Dense(32, activation='relu', kernel_regularizer=l2(0.002))(dropout2)
    
    # Final output with bias terms
    output = Dense(1, activation='linear')(dense3)
    output = Add()([output, user_bias_vec, movie_bias_vec])  # Adding bias terms
    
    # Compile model
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4), loss=tf.keras.losses.Huber(delta=1.0), metrics=["mae"])
    
    return model

# Create & summarize model
model = create_model()
model.summary()

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=50,  # Adjust as needed
    batch_size=128,
    callbacks=[early_stopping]
)
model.save("movie_recommender.h5")


# Function to recommend top N movies for a user
def recommend_movies(user_id, top_n=10):
    if user_id not in ratings["userId"].values:
        print("User ID not found in dataset.")
        return pd.DataFrame()
    
    user_index = user_encoder.transform([user_id])[0]
    movie_indices = np.arange(num_movies)  # Fix: Ensures proper range
    
    predictions = model.predict([np.array([user_index] * num_movies), movie_indices])
    
    top_movie_indices = np.argsort(predictions.flatten())[-top_n:][::-1]
    recommended_movie_ids = movie_encoder.inverse_transform(top_movie_indices)
    
    recommended_movies = movies[movies["movieId"].isin(recommended_movie_ids)]
    return recommended_movies[["movieId", "title"]]

# Example Prediction
user_id = 3  # Replace with actual user ID
movie_id = 26409  # Replace with actual movie ID

if user_id in ratings["userId"].values and movie_id in ratings["movieId"].values:
    user_index = user_encoder.transform([user_id])[0]
    movie_index = movie_encoder.transform([movie_id])[0]

    predicted_rating = model.predict([np.array([user_index]), np.array([movie_index])])
    print(f"Predicted rating for User {user_id} and Movie {movie_id}: {predicted_rating[0][0]:.2f}")

    actual_rating = ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)]['rating'].values
    if len(actual_rating) > 0:
        print(f"Actual rating given by user: {actual_rating[0]}")
    else:
        print("No actual rating found for this user-movie pair.")
else:
    print("User ID or Movie ID not found in dataset.")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

print(" Starting model building...")

# Read dataset
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, left_on='title', right_on='title')

# Clean data
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')
movies['tagline'] = movies['tagline'].fillna('')
movies['cast'] = movies['cast'].fillna('')
movies['crew'] = movies['crew'].fillna('')

# Create a tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tagline'] + movies['cast'] + movies['crew']

# Vectorization
print("🧩 Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vector = vectorizer.fit_transform(movies['tags']).toarray()

# Similarity matrix
print("🔍 Computing similarity matrix...")
similarity = cosine_similarity(vector)

# Final data
print("📦 Preparing final data...")
final_data = movies[['title', 'id']].reset_index(drop=True)

# Save model
print(" Saving model and data...")
joblib.dump((final_data, similarity), 'recommendation_model.pkl')

print(" Model built and saved as 'recommendation_model.pkl'.")

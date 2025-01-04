from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the dataset
anime_df = pd.read_csv('anime_cleaned.csv')  # Replace with the actual file path
ratings_df = pd.read_csv('rating.csv')  # Replace with the actual file path

# Preprocess the data
anime_df = anime_df.dropna(subset=['genre', 'type', 'episodes', 'synopsis', 'studio'])
ratings_df = ratings_df.dropna()

# Filter out anime with very few ratings
min_ratings = 100
anime_stats = ratings_df.groupby('anime_id').size()
popular_anime = anime_stats[anime_stats >= min_ratings].index
ratings_df = ratings_df[ratings_df['anime_id'].isin(popular_anime)]

# User Authentication
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Collaborative Filtering: Neural Collaborative Filtering (NCF)
def build_ncf_model(num_users, num_anime, embedding_size=50):
    user_input = Input(shape=(1,))
    anime_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    anime_embedding = Embedding(num_anime, embedding_size)(anime_input)
    user_vec = Flatten()(user_embedding)
    anime_vec = Flatten()(anime_embedding)
    concat = Concatenate()([user_vec, anime_vec])
    dense = Dense(128, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[user_input, anime_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Content-Based Filtering: Genre, Synopsis, and Studio Similarity
def content_based_filtering(anime_df):
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(anime_df['genre'])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    synopsis_embeddings = model.encode(anime_df['synopsis'].tolist())
    studio_matrix = pd.get_dummies(anime_df['studio']).values
    combined_similarity = (
        0.4 * cosine_similarity(genre_matrix) +
        0.4 * cosine_similarity(synopsis_embeddings) +
        0.2 * cosine_similarity(studio_matrix)
    )
    return combined_similarity

# Hybrid Recommendation
def hybrid_recommendations(user_id, anime_id, ncf_model, anime_similarity, n=5):
    # NCF score
    ncf_score = ncf_model.predict([np.array([user_id]), np.array([anime_id])])[0][0]

    # Content-based filtering score
    anime_idx = anime_df[anime_df['anime_id'] == anime_id].index[0]
    similar_anime_indices = anime_similarity[anime_idx].argsort()[::-1][1:n+1]
    cb_scores = anime_similarity[anime_idx, similar_anime_indices]

    # Combine scores
    hybrid_scores = 0.7 * ncf_score + 0.3 * np.mean(cb_scores)
    return hybrid_scores

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form['user_id']
    user = User(user_id)
    login_user(user)
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    user_id = current_user.id
    anime_id = int(request.form['anime_id'])
    ncf_model = build_ncf_model(num_users=1000, num_anime=1000)  # Replace with actual numbers
    anime_similarity = content_based_filtering(anime_df)
    score = hybrid_recommendations(user_id, anime_id, ncf_model, anime_similarity)
    anime_title = anime_df[anime_df['anime_id'] == anime_id]['name'].values[0]
    return render_template('recommendations.html', user_id=user_id, anime_title=anime_title, score=score)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '').lower()
    results = anime_df[anime_df['name'].str.lower().str.contains(query)].head(10)
    return jsonify(results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
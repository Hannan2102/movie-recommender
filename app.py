import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Set page config
st.set_page_config(page_title="🎥 Movie Recommender", layout="wide")

# ===== Custom CSS Styling =====
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #000;
        color: white;
    }

    .stButton > button {
        background-color: #a855f7;
        color: black;
        font-weight: 600;
        padding: 0.5em 1.2em;
        border-radius: 6px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #9333ea;
        color: black;
    }

    .stSelectbox > div > div {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Title =====
st.markdown("""
    <h1 style='text-align: center;'> <span style="color:white">Movie Recommendation System</span></h1>
    <p style='text-align: center; color:white'>Select your favorite movie and explore similar ones. Click on any recommendation to see full details!</p>
""", unsafe_allow_html=True)

# ===== Load Data =====
movies = pd.read_csv('tmdb_5000_movies.csv')
movies['overview'] = movies['overview'].fillna('')
movies['tags'] = movies['overview']

# ===== Vectorization =====
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vector = vectorizer.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vector)

# ===== Save Model (Optional) =====
joblib.dump((movies, similarity), 'recommendation_model.pkl')

# ===== TMDB API Key =====
API_KEY = '8a8bf0d73eaa8163dcaeaf0269f0ff02'

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Image"

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    return {
        'title': data.get('title', 'N/A'),
        'overview': data.get('overview', 'No overview available.'),
        'release_date': data.get('release_date', 'N/A'),
        'rating': data.get('vote_average', 'N/A'),
        'poster': f"https://image.tmdb.org/t/p/w500/{data.get('poster_path')}" if data.get('poster_path') else "https://via.placeholder.com/500x750?text=No+Image",
        'runtime': data.get('runtime', 'N/A'),
        'tagline': data.get('tagline', ''),
        'genres': ', '.join([genre['name'] for genre in data.get('genres', [])]),
        'languages': ', '.join([lang['english_name'] for lang in data.get('spoken_languages', [])]),
        'votes': data.get('vote_count', 'N/A'),
        'popularity': data.get('popularity', 'N/A'),
        'production': ', '.join([prod['name'] for prod in data.get('production_companies', [])]),
        'trailer': fetch_trailer(movie_id)
    }

@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    for video in data.get('results', []):
        if video['site'] == 'YouTube' and video['type'] == 'Trailer':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

@st.cache_data(show_spinner=False)
def fetch_cast(movie_id, top_n=5):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    cast_list = []
    if 'cast' in data:
        for member in data['cast'][:top_n]:
            cast_list.append({
                'name': member.get('name', 'N/A'),
                'character': member.get('character', 'N/A'),
                'profile': f"https://image.tmdb.org/t/p/w500/{member.get('profile_path')}" if member.get('profile_path') else "https://via.placeholder.com/500x750?text=No+Image"
            })
    return cast_list

def recommend(movie_title):
    try:
        index = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return [], [], []
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    recommended_posters = []
    recommended_ids = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
        recommended_ids.append(movie_id)
    return recommended_movies, recommended_posters, recommended_ids

# ===== Session State =====
if 'clicked_movie' not in st.session_state:
    st.session_state.clicked_movie = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# ===== Movie Selector =====
selected_movie = st.selectbox("🎬 type/select a movie from the list:", movies['title'].sort_values().unique())

# ===== Recommend Button =====
if st.button('Recommend'):
    with st.spinner('Fetching recommendations...'):
        names, posters, ids = recommend(selected_movie)
        st.session_state.recommendations = (names, posters, ids)
        st.session_state.clicked_movie = None

# ===== Show Recommendations =====
if st.session_state.recommendations:
    names, posters, ids = st.session_state.recommendations

    selected_movie_id = movies[movies['title'] == selected_movie].iloc[0].id
    selected_movie_details = fetch_movie_details(selected_movie_id)
    st.markdown("## 🎬 Selected Movie")
    st.image(selected_movie_details['poster'], width=250)
    st.markdown(f"### {selected_movie_details['title']}")
    st.markdown(f"**Overview:** {selected_movie_details['overview'][:200]}...")

    st.markdown("###  Recommended Movies Based on Selected Movie")
    st.markdown("Click on a movie title to see details.")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.image(posters[idx], use_container_width=True)
            if col.button(names[idx], key=f"recommend_{idx}"):
                st.session_state.clicked_movie = ids[idx]

# ===== Movie Details Section =====
if st.session_state.clicked_movie is not None:
    movie_details = fetch_movie_details(st.session_state.clicked_movie)
    cast = fetch_cast(st.session_state.clicked_movie)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"## {movie_details['title']}")
    st.image(movie_details['poster'], use_column_width=False, width=300)
    st.markdown(f"**Tagline:** {movie_details['tagline']}")
    st.markdown(f"**Genres:** {movie_details['genres']}")
    st.markdown(f"**Rating:** {movie_details['rating']}  ⭐")
    st.markdown(f"**Release Date:** {movie_details['release_date']}")
    st.markdown(f"**Runtime:** {movie_details['runtime']} minutes")
    st.markdown(f"**Popularity:** {movie_details['popularity']}")
    st.markdown(f"**Votes:** {movie_details['votes']}")
    st.markdown(f"**Languages:** {movie_details['languages']}")
    st.markdown(f"**Production Companies:** {movie_details['production']}")
    st.markdown(f"**Overview:** {movie_details['overview']}")

    if movie_details['trailer']:
        st.markdown(f"[▶️ Watch Trailer]({movie_details['trailer']})", unsafe_allow_html=True)

    if cast:
        st.markdown("### Cast")
        cast_cols = st.columns(len(cast))
        for idx, cast_col in enumerate(cast_cols):
            with cast_col:
                st.image(cast[idx]['profile'], use_container_width=True)
                st.markdown(f"**{cast[idx]['name']}**")
                st.markdown(f"*as {cast[idx]['character']}*")

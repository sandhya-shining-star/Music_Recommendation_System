"""
app.py
------
A polished, animated Streamlit front-end for the lyrics-based TF-IDF
music recommendation system.

Highlights
- Mood-aware theming: each song is scanned for "vibe" keywords (love,
  party, sad, chill, angry, dance...) and the whole page (gradient
  background + accent color + floating emojis) shifts to match it.
- Animated floating emoji background, glassmorphism song cards,
  similarity bars, lyric snippets, downloadable playlist, shuffle button.
- Graceful fallback to a small built-in demo dataset if the trained
  artifacts (artifacts/songs.pkl etc., produced by prepare_data.py)
  aren't present yet, so the app always runs out of the box.

Run:
    streamlit run app.py
"""

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="LyricMatch — AI Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = "artifacts"

# --------------------------------------------------------------------------
# MOOD ENGINE — keyword based "vibe" detector + theme palettes
# --------------------------------------------------------------------------
MOODS = {
    "love": {
        "keywords": ["love", "heart", "kiss", "baby", "forever", "darling", "romance"],
        "emojis": ["💖", "💕", "🌹", "💘", "😍"],
        "gradient": "linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #fbc2eb 100%)",
        "accent": "#e0457b",
        "label": "Romantic 💕",
    },
    "party": {
        "keywords": ["party", "dance", "club", "night", "jam", "beat", "groove"],
        "emojis": ["🎉", "🪩", "🕺", "💃", "🔥"],
        "gradient": "linear-gradient(135deg, #f6d365 0%, #fda085 50%, #fa709a 100%)",
        "accent": "#ff6f3c",
        "label": "Party 🎉",
    },
    "sad": {
        "keywords": ["cry", "tears", "lonely", "pain", "sad", "broken", "goodbye", "lost"],
        "emojis": ["😢", "💔", "🌧️", "🥀", "😞"],
        "gradient": "linear-gradient(135deg, #485563 0%, #29323c 100%)",
        "accent": "#6c8ebf",
        "label": "Melancholic 😢",
    },
    "chill": {
        "keywords": ["sun", "summer", "ocean", "beach", "sky", "breeze", "calm", "easy"],
        "emojis": ["🌴", "🌊", "☀️", "😌", "🍹"],
        "gradient": "linear-gradient(135deg, #43cea2 0%, #185a9d 100%)",
        "accent": "#1f9d8b",
        "label": "Chill 🌴",
    },
    "angry": {
        "keywords": ["fight", "fire", "hate", "rage", "scream", "war", "blood"],
        "emojis": ["🔥", "😤", "⚡", "🤬", "💢"],
        "gradient": "linear-gradient(135deg, #870000 0%, #190a05 100%)",
        "accent": "#d7263d",
        "label": "Intense 🔥",
    },
    "dreamy": {
        "keywords": ["dream", "star", "moon", "night", "sky", "fly", "magic", "wonder"],
        "emojis": ["✨", "🌙", "⭐", "🪐", "💫"],
        "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "accent": "#8a63d2",
        "label": "Dreamy ✨",
    },
    "default": {
        "keywords": [],
        "emojis": ["🎵", "🎶", "🎧", "🎤", "🎸"],
        "gradient": "linear-gradient(135deg, #1e1e2f 0%, #2b1055 50%, #7597de 100%)",
        "accent": "#7c5cff",
        "label": "Pure Vibes 🎶",
    },
}


def detect_mood(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "default"
    text_low = text.lower()
    scores = {}
    for mood, cfg in MOODS.items():
        if mood == "default":
            continue
        scores[mood] = sum(text_low.count(kw) for kw in cfg["keywords"])
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "default"


# --------------------------------------------------------------------------
# DATA LOADING (cached) — real artifacts if present, else a small demo set
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_data():
    songs_path = os.path.join(ARTIFACT_DIR, "songs.pkl")
    vec_path = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
    mat_path = os.path.join(ARTIFACT_DIR, "tfidf_matrix.pkl")

    if os.path.exists(songs_path) and os.path.exists(vec_path) and os.path.exists(mat_path):
        df = pd.read_pickle(songs_path)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(mat_path, "rb") as f:
            matrix = pickle.load(f)
        return df.reset_index(drop=True), vectorizer, matrix, True

    # ---- fallback demo dataset so the app always runs ----
    demo = pd.DataFrame(
        {
            "artist": ["Skylar Vale", "Skylar Vale", "Midnight Rovers", "Midnight Rovers",
                       "Coral Bloom", "Coral Bloom", "Ember Park", "Ember Park"],
            "song": ["Forever Yours", "Heart on Fire", "City Lights Dance", "Party Til Dawn",
                     "Tears in the Rain", "Empty Room", "Sunset Drive", "Ocean Breeze"],
            "text": [
                "I love you forever my darling, you are my heart and my baby",
                "Kiss me under the stars, my heart beats only for your love",
                "Let's dance all night in the club, feel the beat, feel the groove",
                "Party till dawn, jam to the rhythm, the night is alive",
                "I cry alone, tears falling, lonely nights and broken dreams",
                "Goodbye my love, this empty room reminds me of the pain",
                "Sun on my skin, summer breeze, driving down by the ocean sky",
                "Calm waves, easy days, the beach and sky make me feel free",
            ],
        }
    )
    demo["cleaned_text"] = demo["text"]
    vectorizer = TfidfVectorizer(max_features=2000)
    matrix = vectorizer.fit_transform(demo["cleaned_text"])
    return demo, vectorizer, matrix, False


df, tfidf_vectorizer, tfidf_matrix, USING_REAL_DATA = load_data()


# --------------------------------------------------------------------------
# RECOMMENDATION LOGIC
# --------------------------------------------------------------------------
def recommend_songs(song_name: str, top_n: int = 6):
    matches = df.index[df["song"].str.lower() == song_name.lower()]
    if len(matches) == 0:
        return None, None
    idx = matches[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = np.argsort(sims)[::-1]
    order = [i for i in order if i != idx][:top_n]
    result = df.iloc[order].copy()
    result["similarity"] = sims[order]
    return result, idx


# --------------------------------------------------------------------------
# THEME (mood) STATE
# --------------------------------------------------------------------------
if "current_mood" not in st.session_state:
    st.session_state.current_mood = "default"

mood_cfg = MOODS[st.session_state.current_mood]


# --------------------------------------------------------------------------
# CSS — glassmorphism, animated gradient, floating emojis
# --------------------------------------------------------------------------
floating_emojis_html = "".join(
    f'<div class="floater" style="left:{random.randint(2, 96)}%; '
    f'animation-duration:{random.randint(10, 22)}s; '
    f'animation-delay:-{random.randint(0, 20)}s; font-size:{random.randint(18, 38)}px;">'
    f'{random.choice(mood_cfg["emojis"])}</div>'
    for _ in range(18)
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    .stApp {{
        background: {mood_cfg["gradient"]};
        background-size: 300% 300%;
        animation: gradientShift 18s ease infinite;
        transition: background 1.2s ease-in-out;
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .floater-container {{
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
    }}
    .floater {{
        position: absolute;
        bottom: -10%;
        opacity: 0.55;
        animation-name: floatUp;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
        filter: drop-shadow(0 0 6px rgba(255,255,255,0.25));
    }}
    @keyframes floatUp {{
        0%   {{ transform: translateY(0) rotate(0deg); opacity: 0; }}
        10%  {{ opacity: 0.6; }}
        90%  {{ opacity: 0.5; }}
        100% {{ transform: translateY(-115vh) rotate(360deg); opacity: 0; }}
    }}

    .glass-card {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.25);
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        z-index: 1;
        color: #fff;
    }}
    .glass-card:hover {{
        transform: translateY(-6px) scale(1.012);
        box-shadow: 0 14px 40px rgba(0,0,0,0.3);
    }}

    .hero-title {{
        font-size: 3rem;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 4px 18px rgba(0,0,0,0.35);
        margin-bottom: 0;
    }}
    .hero-sub {{
        font-size: 1.05rem;
        color: rgba(255,255,255,0.85);
        margin-top: 4px;
    }}

    .mood-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        color: #fff;
        font-weight: 600;
        font-size: 0.9rem;
        backdrop-filter: blur(6px);
    }}

    .sim-bar-bg {{
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin-top: 6px;
    }}
    .sim-bar-fill {{
        height: 100%;
        border-radius: 10px;
        background: {mood_cfg["accent"]};
        transition: width 1s ease;
    }}

    .song-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 2px;
    }}
    .song-artist {{
        font-size: 0.95rem;
        color: rgba(255,255,255,0.78);
        margin-bottom: 10px;
    }}
    .lyric-snippet {{
        font-style: italic;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        border-left: 3px solid {mood_cfg["accent"]};
        padding-left: 10px;
        margin-top: 8px;
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
    }}
    section[data-testid="stSidebar"] * {{
        color: #fff !important;
    }}

    div.stButton > button {{
        background: {mood_cfg["accent"]};
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 22px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        filter: brightness(1.08);
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>

    <div class="floater-container">{floating_emojis_html}</div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎧 LyricMatch")
    st.markdown("AI-powered recommendations based on **lyrical similarity** (TF-IDF + cosine similarity).")
    st.markdown("---")
    top_n = st.slider("Number of recommendations", min_value=3, max_value=12, value=6)
    st.markdown("---")
    st.markdown(f"**Dataset size:** {len(df):,} songs")
    st.markdown(f"**Artists:** {df['artist'].nunique():,}")
    if not USING_REAL_DATA:
        st.info(
            "Running on a small **demo dataset**.\n\n"
            "Run `prepare_data.py` on the full Spotify Million Song "
            "Dataset to unlock real recommendations:\n\n"
            "`python prepare_data.py --csv spotify_millsongdata.csv`"
        )
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit, scikit-learn & TF-IDF")

# --------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="hero-title">🎶 LyricMatch</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Discover songs with the same soul — powered by lyric analysis.</p>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        f'<div style="text-align:right; margin-top:18px;">'
        f'<span class="mood-badge">Current vibe: {mood_cfg["label"]}</span></div>',
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------
# SEARCH CONTROLS
# --------------------------------------------------------------------------
song_list = sorted(df["song"].dropna().unique().tolist())

c1, c2 = st.columns([4, 1])
with c1:
    selected_song = st.selectbox("🔍 Pick a song you love", options=song_list, index=0)
with c2:
    st.write("")
    st.write("")
    shuffle = st.button("🎲 Surprise me")

if shuffle:
    selected_song = random.choice(song_list)
    st.toast(f"Shuffled to: {selected_song} 🎵")

go = st.button("✨ Find similar songs", type="primary")

# --------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------
if go or shuffle:
    with st.spinner("Analyzing lyrics and matching vibes... 🎼"):
        time.sleep(0.4)
        recs, src_idx = recommend_songs(selected_song, top_n=top_n)

    if recs is None:
        st.error("Hmm, couldn't find that song in the dataset. Try another one!")
    else:
        src_text = df.loc[src_idx, "text"] if "text" in df.columns else ""
        new_mood = detect_mood(src_text)
        if new_mood != st.session_state.current_mood:
            st.session_state.current_mood = new_mood
            st.rerun()

        src_row = df.loc[src_idx]
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="song-title">🎤 Seed track: {src_row['song']}</div>
                <div class="song-artist">by {src_row['artist']}</div>
                <span class="mood-badge">{mood_cfg['label']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"### {random.choice(mood_cfg['emojis'])} Recommended for you")

        cols = st.columns(2)
        for i, (_, row) in enumerate(recs.iterrows()):
            sim_pct = round(float(row["similarity"]) * 100, 1)
            emoji = random.choice(mood_cfg["emojis"])
            snippet = ""
            if "text" in row and isinstance(row["text"], str):
                snippet = row["text"].strip().replace("\n", " ")[:140] + "..."
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="song-title">{emoji} {row['song']}</div>
                        <div class="song-artist">by {row['artist']}</div>
                        <div class="lyric-snippet">"{snippet}"</div>
                        <div style="margin-top:12px; font-size:0.85rem; color:rgba(255,255,255,0.85);">
                            Match score: {sim_pct}%
                        </div>
                        <div class="sim-bar-bg">
                            <div class="sim-bar-fill" style="width:{sim_pct}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        playlist_text = "\n".join(f"{r['artist']} — {r['song']}" for _, r in recs.iterrows())
        st.download_button(
            "⬇️ Download playlist as .txt",
            data=playlist_text,
            file_name=f"{selected_song.replace(' ', '_')}_playlist.txt",
            mime="text/plain",
        )
else:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:40px;">
            <h3>👋 Pick a song above and hit <i>Find similar songs</i></h3>
            <p>The background, emojis, and color theme will shift to match the mood of your music.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
"""
prepare_data.py
----------------
Run this ONCE to turn the raw Spotify Million Song dataset CSV into the
artifacts (cleaned dataframe, TF-IDF vectorizer, TF-IDF matrix) that
app.py loads. This mirrors the preprocessing done in the original
notebook (Music_Recommendation_System_using_Python_code_prep_2.ipynb).

Usage:
    python prepare_data.py --csv spotify_millsongdata.csv --sample 10000

Outputs (written to ./artifacts/):
    songs.pkl            -> cleaned dataframe (artist, song, text, cleaned_text)
    tfidf_vectorizer.pkl -> fitted TfidfVectorizer
    tfidf_matrix.pkl      -> sparse TF-IDF matrix (scipy sparse, saved with pickle)
"""

import os
import re
import argparse
import pickle

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def download_nltk_data():
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def preprocess_text(text, stop_words):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to spotify_millsongdata.csv")
    parser.add_argument("--sample", type=int, default=10000, help="Number of songs to sample (use 0 for full dataset)")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--max_features", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Downloading NLTK resources...")
    download_nltk_data()
    stop_words = set(stopwords.words("english"))

    print(f"Loading dataset from {args.csv} ...")
    df = pd.read_csv(args.csv)

    if args.sample and args.sample > 0 and args.sample < len(df):
        df = df.sample(args.sample, random_state=42)

    if "link" in df.columns:
        df = df.drop("link", axis=1)
    df = df.dropna(subset=["text", "song", "artist"]).reset_index(drop=True)

    print("Cleaning lyrics (this can take a few minutes)...")
    df["cleaned_text"] = df["text"].apply(lambda t: preprocess_text(t, stop_words))

    print("Fitting TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=args.max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

    print("Saving artifacts...")
    df.to_pickle(os.path.join(args.out_dir, "songs.pkl"))
    with open(os.path.join(args.out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(args.out_dir, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print(f"Done! Artifacts saved in '{args.out_dir}/'. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
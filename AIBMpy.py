import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import joblib
import streamlit as st

# Load datasets
df1 = pd.read_csv("C:/Users/Sukkiiii/Downloads/Audible_Catlog.csv")
df2 = pd.read_csv("C:/Users/Sukkiiii/Downloads//Audible_Catlog_Advanced_Features.csv")

# Merge datasets on 'Book Name' and 'Author'
df = pd.merge(df1, df2, on=['Book Name', 'Author'], how='outer')

# Drop duplicate columns (since some fields appear twice after merging)
df = df.loc[:, ~df.columns.duplicated()]

# Handling missing values
df.fillna({'Rating_x': df['Rating_x'].median(),'Rating_y': df['Rating_y'].median(), 'Number of Reviews_x': 0,'Number of Reviews_y': 0, 'Price_x': df['Price_x'].median(),'Price_y': df['Price_y'].median()}, inplace=True)
df.dropna(subset=['Book Name', 'Author'], inplace=True)  # Ensure book names and authors are present

# Standardizing text fields
df['Book Name'] = df['Book Name'].str.strip()
df['Author'] = df['Author'].str.strip()
df['Description'] = df['Description'].fillna('No description available.')

def convert_time_to_minutes(time_str):
    """Convert time format (e.g., '5 hours and 30 minutes') to minutes."""
    if pd.isna(time_str):
        return np.nan
    time_str = time_str.lower()
    hours = int(time_str.split("hours")[0]) if "hours" in time_str else 0
    minutes = int(time_str.split("minutes")[0].split()[-1]) if "minutes" in time_str else 0
    return hours * 60 + minutes

df['Listening Time (minutes)'] = df['Listening Time'].apply(convert_time_to_minutes)

df['Main Genre'] = df['Ranks and Genre'].apply(lambda x: ', '.join([genre.split('in ')[-1].strip() for genre in x.split(',') if 'in ' in genre]) if pd.notna(x) else 'Unknown')

# Drop redundant columns
df.drop(columns=['Ranks and Genre', 'Listening Time'], inplace=True)

df['Rating']=(df['Rating_x']+df['Rating_y'])/2

df['Price']=(df['Price_x']+df['Price_y'])/2

df['Number of Reviews']=(df['Number of Reviews_x']+df['Number of Reviews_y'])/2

df.drop(columns=['Rating_x', 'Number of Reviews_x','Price_x','Rating_y', 'Number of Reviews_y','Price_y'], inplace=True)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
df['Description'] = df['Description'].fillna('')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

# Compute similarity scores
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Content-Based Filtering - Function to get book recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df.index[df['Book Name'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return df[['Book Name', 'Author']].iloc[book_indices]

# Clustering Books using K-Means
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(tfidf_matrix)

# Save the clustering model
joblib.dump(kmeans, "kmeans_model.pkl")

# Collaborative Filtering with Nearest Neighbors
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(tfidf_matrix)

# Save trained KNN model
joblib.dump(knn, "knn_model.pkl")

# Function to get collaborative filtering recommendations
def get_knn_recommendations(title):
    idx = df.index[df['Book Name'] == title].tolist()[0]
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=6)
    book_indices = indices.flatten()[1:]
    return df[['Book Name', 'Author']].iloc[book_indices]

# Streamlit App
st.title("Book Recommendation System")
user_input = st.text_input("Enter a book title:")

if st.button("Get Recommendations"):
    if user_input in df['Book Name'].values:
        st.write("### Content-Based Recommendations:")
        st.dataframe(get_recommendations(user_input))
        st.write("### Collaborative Filtering Recommendations:")
        st.dataframe(get_knn_recommendations(user_input))
    else:
        st.write("Book not found. Try another title.")

# Save the cleaned dataset
df.to_csv("C:/Users/Sukkiiii/Desktop/ME_DATA/DS/cleaned_audible_data.csv", index=False)

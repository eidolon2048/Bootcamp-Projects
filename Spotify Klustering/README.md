# Spotify Song Clustering

This is an exploratory machine learning project that groups Spotify songs into clusters using audio features. The original work is contained in the notebook `ML_spotify_5000_online.ipynb`, and a cleaner improved version is available in `spotify_song_clustering_improved.ipynb`.

The goal is to use unsupervised learning to find groups of songs with similar musical characteristics. This can be useful as a starting point for playlist creation, music discovery, or simple recommendation logic.

## Dataset

The notebook loads a Spotify dataset of roughly 5,000 songs from a Google Drive CSV link. Each row represents a track and includes artist/title information plus Spotify audio features such as:

- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- valence
- tempo

Some non-feature columns are removed before modelling, including track identifiers and metadata fields. The notebook also drops `mode` and `liveness` before clustering.

## Method

The notebook follows this workflow:

1. Import the dataset from Google Drive.
2. Clean column names by removing whitespace.
3. Drop metadata and selected audio columns.
4. Keep artist and song title as identifiers.
5. Scale the remaining numeric audio features with `MinMaxScaler`.
6. Test different KMeans cluster counts using inertia and silhouette score.
7. Train a final KMeans model with 36 clusters.
8. Inspect cluster sizes, distances between cluster centers, and sample songs from each cluster.

## Key Result

The final notebook uses `k=36` clusters. This choice was based on the notebook's exploration of inertia and silhouette score, with extra attention to cluster counts above 20 because the project aimed to keep clusters reasonably small.

The saved notebook output shows that most clusters stay below the target size of 250 songs. One cluster contains 260 songs, so the size target is almost met but not fully satisfied.

## Limitations

- KMeans results are not fully reproducible because most model runs do not set a fixed `random_state`.
- The dataset is loaded from an external Google Drive link, so the notebook depends on that link remaining available.
- The final cluster count is selected visually from plots rather than from a clear automated rule.
- One cluster exceeds the intended maximum size of 250 songs.
- The project is exploratory and does not include a production recommender, app, or formal test suite.

## Possible Improvements

- Set `random_state` and `n_init` explicitly for reproducible KMeans results.
- Choose the final cluster count using both silhouette score and the maximum-cluster-size requirement.
- Compare different feature selections, especially whether dropping `mode` and `liveness` improves the result.
- Add cluster summaries that describe the average audio profile of each group.
- Select representative songs nearest to each cluster center instead of only showing random samples.
- Save the dataset locally or document how to recreate it if the Google Drive link changes.

## How to Run

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open `spotify_song_clustering_improved.ipynb` and run the cells from top to bottom.

The original notebook, `ML_spotify_5000_online.ipynb`, is kept unchanged as a reference.

## Project Files

- `ML_spotify_5000_online.ipynb` - original exploratory clustering notebook kept as reference
- `spotify_song_clustering_improved.ipynb` - cleaner improved notebook with reproducible KMeans evaluation
- `README.md` - project overview and usage notes
- `requirements.txt` - Python dependencies used by the notebook
- `.gitignore` - common local files to ignore

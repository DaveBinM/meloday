#!/usr/bin/env bash
# Download the optional Essentia high-level classification heads into this folder.
# They run on the EffNet/MusiCNN embeddings Meloday already extracts (the big embedding
# models discogs-effnet-bs64-1.pb / msd-musicnn-1.pb must already be here). Each head is ~3 MB.
set -e
cd "$(dirname "$0")"
BASE="https://essentia.upf.edu/models/classification-heads"
dl() {  # task  basename
  for ext in pb json; do
    echo "  $2.$ext"
    curl -fsSL "$BASE/$1/$2.$ext" -o "$2.$ext"
  done
}
echo "Downloading high-level heads into $(pwd) ..."
dl mtg_jamendo_moodtheme mtg_jamendo_moodtheme-discogs-effnet-1
dl genre_discogs400      genre_discogs400-discogs-effnet-1
dl danceability          danceability-msd-musicnn-1
for m in mood_happy mood_sad mood_aggressive mood_relaxed mood_party mood_acoustic mood_electronic; do
  dl "$m" "$m-msd-musicnn-1"
done
echo "Done. $(ls -1 *-msd-musicnn-1.pb *-discogs-effnet-1.pb 2>/dev/null | wc -l) head .pb files present."

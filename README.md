# NFL Combine Slant Explorer — bundled data

## Deploy
1. Extract this ZIP.
2. Upload app.py, nfl_slant_starter.py, requirements.txt, README.md, the complete data folder, and .streamlit/config.toml to your GitHub repository. Keep the data folder structure intact.
3. In Streamlit Community Cloud select the repository, branch, and app.py, then deploy.
4. Data loads automatically; nobody needs to upload CSVs in the app.

Keep the repository and app access limited to your authorized competition team and follow Kaggle competition sharing terms. Source data is included in this package.

## Included datasets
players.csv, combine_results.csv, combine_tracking.csv, player_play.csv, and game_tracking_2023.csv.
The explorer currently uses players and combine tracking only. Other bundled datasets are available for future analysis. games.csv, player_career_successes.csv, and game_tracking_2024/2025.csv were not supplied and are not included.

Data files are gzip compressed and split into chunks under 20 MB for repository uploads. Do not extract, rename, or open individual parts; the app reassembles them. The complete data folder must be committed. Large browser uploads can take time; upload the data folder in batches if necessary.

## Local use
python -m pip install -r requirements.txt
python -m streamlit run app.py

## Limitations
Candidate turn detection uses peak smoothed heading-change rate while moving >=3 yd/s with 0.3 seconds of tracking on each side. Entry/exit speeds average adjacent 0.3-second windows. Markers require visual validation. This is exploratory; there is no fitted NFL prediction model. Hosted memory requirements and the Streamlit UI have not been tested here. The loader currently reads only the two datasets required by the explorer.

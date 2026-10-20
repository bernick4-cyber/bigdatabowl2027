# NFL Combine Slant Explorer

A Streamlit starter app for inspecting WR combine slant paths and exploratory speed-through-turn measurements. No NFL prediction model is included yet.

## Deploy from GitHub to Streamlit
1. Create a repository and upload the extracted project files. Keep `app.py`, `nfl_slant_starter.py`, and `requirements.txt` at the repository root, and retain `.streamlit/config.toml` in its folder.
2. In Streamlit Community Cloud, create an app from the repository. Select your branch and set the main file to `app.py`.
3. Deploy and open the app. Upload `players.csv` and `combine_tracking.csv` (or the copy with `(1)` in its name) in the sidebar.
4. Choose a receiver and an attempt to inspect the route and speed chart. Download the selected measurements as CSV.

Do not upload the ZIP itself as your project source: extract it first. If GitHub's upload picker hides dotfiles, create `.streamlit/config.toml` with GitHub's Add file > Create new file feature and paste its contents.

## Local use
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data and access
Competition CSVs are not included. Each teammate obtains data through Kaggle under the competition terms. Confirm those terms permit the intended hosted processing and sharing before deployment. Configure app access for your team; a private GitHub repository alone does not establish private app access. Uploads are processed on the Streamlit host and may use substantial RAM; the 500 MB upload setting does not guarantee sufficient hosting memory. This app only needs combine tracking and players, not the much larger NFL game-tracking files.

## Measurement limitations
The candidate turn is the peak absolute heading-change rate after circular unwrapping and three-frame smoothing, restricted to speed >=3 yards/second and at least 0.3 seconds from either endpoint. Entry and exit speed are averages over adjacent 0.3-second windows. Visually review markers: peak turning can reflect another movement rather than the intended cut. This heuristic has not been validated, and no relationship with NFL separation has been established.

`nfl_slant_starter.py` also supports batch plotting and CSV export; see its docstring for commands.

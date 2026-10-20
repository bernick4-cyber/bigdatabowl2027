"""Streamlit entry point for the NFL combine slant explorer."""
from pathlib import Path
import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from nfl_slant_starter import inspect_attempt

st.set_page_config(page_title='NFL Combine Slant Explorer', layout='wide')
st.title('NFL Combine Slant Explorer')
st.write('Inspect receiver movement in combine slants before testing its relationship with NFL separation.')
st.info('Exploratory turn detection: red markers identify a candidate turn using direction changes. These measurements have not been validated or shown to predict NFL performance.')

@st.cache_data(show_spinner=False)
def load_csv(raw):
    return pd.read_csv(io.BytesIO(raw))

with st.sidebar:
    st.header('Load competition data')
    st.caption('Files stay out of the GitHub repository. Upload your Kaggle downloads here.')
    players_file = st.file_uploader('players.csv', type=['csv'], key='players')
    tracking_file = st.file_uploader('combine_tracking.csv', type=['csv'], key='tracking')
    st.caption('Tracking upload limit: 500 MB per file. Uploaded data is processed by the hosted app.')
if not players_file or not tracking_file:
    st.write('Upload both files in the sidebar to begin. The tracking filename may include “(1)”.')
    st.stop()
try:
    with st.spinner('Reading CSV files…'):
        players = load_csv(players_file.getvalue())
        tracking = load_csv(tracking_file.getvalue())
    for label, df, required in [
        ('players', players, {'nfl_id','display_name','nfl_position'}),
        ('tracking', tracking, {'nfl_id','event_id','entity_type','drill_name','time','x','y','s','dir'})]:
        missing = required - set(df.columns)
        if missing:
            st.error(f'{label} file is missing columns: {", ".join(sorted(missing))}')
            st.stop()
    wr = players[players.nfl_position.eq('WR')].drop_duplicates('nfl_id')
    data = tracking[tracking.nfl_id.isin(wr.nfl_id) & tracking.entity_type.eq('PLAYER') & tracking.drill_name.eq('SLANT_ROUTE_LEFT')].copy()
    if data.empty:
        st.error('No WR SLANT_ROUTE_LEFT player frames were found.'); st.stop()
    if data.duplicated(['event_id','nfl_id','time']).any():
        st.error('Duplicate tracking frames found. Resolve them before measuring.'); st.stop()
    if data[['x','y','s','dir','time']].isna().any().any():
        st.error('Missing tracking measurements found. Resolve them before measuring.'); st.stop()
    names = wr.set_index('nfl_id').display_name.to_dict()
    left, right = st.columns(2)
    left.metric('Receivers with slant tracking', data.nfl_id.nunique())
    right.metric('Drill attempts', data[['event_id','nfl_id']].drop_duplicates().shape[0])
    player = st.selectbox('Receiver', sorted(data.nfl_id.unique(), key=lambda n: str(names[n])), format_func=lambda n: f'{names[n]} ({int(n)})')
    subset = data[data.nfl_id.eq(player)]
    event = st.selectbox('Drill attempt (event ID)', subset.event_id.unique())
    g, i, metrics = inspect_attempt(subset[subset.event_id.eq(event)])
    if metrics is None:
        st.warning('This attempt does not have enough usable moving frames for the candidate turn measurement.'); st.stop()
    g = g.reset_index(drop=True)
    seconds = (g.time - g.time.iloc[0]).dt.total_seconds()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    x, y = g.x-g.x.iloc[0], g.y-g.y.iloc[0]
    axs[0].plot(x,y,'-o',markersize=3)
    axs[0].scatter(x.iloc[i],y.iloc[i],c='red',label='Candidate turn',zorder=3)
    axs[0].set(xlabel='x displacement (yards)',ylabel='y displacement (yards)',title='Route path')
    axs[0].set_aspect('equal'); axs[0].legend()
    axs[1].plot(seconds,g.s)
    axs[1].axvline(seconds.iloc[i],color='red',label='Candidate turn')
    axs[1].set(xlabel='Time since first frame (seconds)',ylabel='Speed (yards/second)',title='Speed over time'); axs[1].legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption('Coordinates are relative to the first frame. Routes are not rotated or aligned across players.')
    c1,c2,c3 = st.columns(3)
    c1.metric('Entry speed',f'{metrics["entry_speed_yd_s"]:.2f} yd/s')
    c2.metric('Exit speed',f'{metrics["exit_speed_yd_s"]:.2f} yd/s')
    c3.metric('Exit / entry speed',f'{metrics["exit_entry_speed_ratio"]:.2f}')
    st.caption('Entry and exit use the 0.3-second windows before and after the candidate turn. A ratio above 1 means the player accelerated across these windows; it is not proof of better route running.')
    result = pd.DataFrame([dict(nfl_id=int(player),display_name=names[player],event_id=event,**metrics)])
    st.download_button('Download selected attempt measurements',result.to_csv(index=False),'slant_attempt_metrics.csv','text/csv')
    with st.expander('View tracking frames'):
        st.dataframe(g,hide_index=True)
except (ValueError, KeyError, pd.errors.ParserError) as exc:
    st.error(f'Could not process these files: {exc}')

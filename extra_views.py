import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from data_access import manifest, read_data, select_tracking
@st.cache_data(max_entries=3)
def table(name): return read_data(name)
@st.cache_data(max_entries=3)
def frames(name, game, player): return select_tracking(name,game,player)
def show(page):
    available=manifest()
    if page=='Dataset inventory':
        expected=['players.csv','combine_results.csv','combine_tracking.csv','player_play.csv','games.csv','player_career_successes.csv']+[f'game_tracking_{y}.csv' for y in (2023,2024,2025)]
        st.dataframe(pd.DataFrame([{'Dataset':n,'Status':'Included' if n in available else 'Not yet packaged'} for n in expected]),hide_index=True)
        return
    if page=='Player profiles':
        p=table('players.csv'); name=st.selectbox('Player',p.nfl_id,format_func=lambda n:p.set_index('nfl_id').loc[n,'display_name'])
        for filename in ['players.csv','combine_results.csv','player_career_successes.csv']:
            st.subheader(filename)
            if filename in available:
                d=table(filename);st.dataframe(d[d.nfl_id.eq(name)],hide_index=True)
            else:st.info('Dataset has not been packaged yet.')
        st.caption('Career totals reflect different amounts of experience across draft classes.')
        return
    if 'games.csv' not in available:
        st.info('Add games.csv with the preparation helper first.');return
    games=table('games.csv');season=st.selectbox('Season',sorted(games.season.unique()))
    filename=f'game_tracking_{season}.csv'
    if filename not in available:
        st.info(f'{filename} is not packaged yet. Run prepare_remaining_data.py on your computer.');return
    g=games[games.season.eq(season)]; game=st.selectbox('Game',g.game_id,format_func=lambda n:f"{n} — {g.set_index('game_id').loc[n,'visitor_team_abbr']} at {g.set_index('game_id').loc[n,'home_team_abbr']}")
    plays=table('player_play.csv'); eligible=plays[plays.game_id.eq(game)]
    names=table('players.csv').set_index('nfl_id').display_name.to_dict()
    if eligible.empty:st.info('No cohort player-play records for this game.');return
    player=st.selectbox('Player',sorted(eligible.nfl_id.unique()),format_func=lambda n:names.get(n,str(n)))
    if st.button('Load player tracking for this game'):
        with st.spinner('Scanning season in chunks. The first request may take several minutes…'):
            st.session_state['tracking_selection']=(filename,int(game),int(player),frames(filename,int(game),int(player)))
    selection=st.session_state.get('tracking_selection')
    if selection is None or selection[:3]!=(filename,int(game),int(player)):return
    f=selection[3]
    if f.empty:st.info('No tracking frames for this selection.');return
    play=st.selectbox('Play',sorted(f.play_id.unique())); q=f[f.play_id.eq(play)].sort_values('time')
    fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,120),ylim=(0,53.33),xlabel='Field x (yards)',ylabel='Field y (yards)')
    for x in range(10,111,10):ax.axvline(x,color='gray',alpha=.25)
    ax.plot(q.x,q.y,'-o',markersize=2);ax.set_aspect('equal');st.pyplot(fig);plt.close(fig)
    st.caption('Only the selected cohort player is shown. Coordinates are not normalized for play direction.')
    st.dataframe(q,hide_index=True)
    st.download_button('Download selected play',q.to_csv(index=False),'selected_play.csv','text/csv')

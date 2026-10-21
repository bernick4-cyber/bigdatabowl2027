import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from data_access import manifest, read_data, select_tracking
from presentation import detail_table, friendly

@st.cache_data(max_entries=3)
def table(name): return read_data(name)
@st.cache_data(max_entries=3)
def frames(name, game, player): return select_tracking(name,game,player)

def show(page):
    available=manifest()
    if page=='Data coverage':
        st.title('Research coverage')
        st.write('Check which seasons and player information are available to the team.')
        labels={'players.csv':'Player directory','combine_results.csv':'Combine measurements','combine_tracking.csv':'Combine movement','player_play.csv':'NFL play performance','games.csv':'Schedule and matchups','player_career_successes.csv':'Career playing time and awards',**{f'game_tracking_{y}.csv':f'{y} NFL movement' for y in (2023,2024,2025)}}
        st.metric('Available research sources',f'{sum(n in available for n in labels)} of {len(labels)}')
        st.dataframe(pd.DataFrame([{'Research source':label,'Availability':'Ready' if n in available else 'Not available yet'} for n,label in labels.items()]),hide_index=True,use_container_width=True)
        return
    if page=='Player profiles':
        st.title('Player profiles')
        st.write('Explore draft background, combine measurements, and career playing time.')
        p=table('players.csv').drop_duplicates('nfl_id');names=p.set_index('nfl_id').display_name.to_dict()
        pos=st.selectbox('Position',['All positions']+sorted(p.nfl_position.dropna().unique()))
        choices=p if pos=='All positions' else p[p.nfl_position.eq(pos)]
        player=st.selectbox('Player',sorted(choices.nfl_id,key=lambda n:str(names[n])),format_func=lambda n:names[n])
        row=p[p.nfl_id.eq(player)].iloc[0]
        st.subheader(names[player]);st.caption(f"{row.nfl_position} · {row.college_name} · {int(row.draft_year)} draft class")
        tabs=st.tabs(['Background','Combine results','Career summary'])
        for tab,filename in zip(tabs,['players.csv','combine_results.csv','player_career_successes.csv']):
            with tab:
                if filename not in available: st.info('This information is not available yet.');continue
                d=table(filename);chosen=d[d.nfl_id.eq(player)]
                if chosen.empty:st.info('No measurements recorded for this player.')
                else:detail_table(chosen)
        st.caption('Career totals span different amounts of experience across draft classes. A blank measurement means it was not recorded.')
        return
    st.title('NFL movement explorer')
    st.write('Choose a matchup and player to inspect their path on an individual play.')
    if 'games.csv' not in available:st.info('The game schedule is not available yet.');return
    games=table('games.csv');season=st.selectbox('Season',sorted(games.season.unique()))
    filename=f'game_tracking_{season}.csv'
    if filename not in available:st.info('Movement tracking for this season is not available yet.');return
    g=games[games.season.eq(season)];lookup=g.set_index('game_id')
    def game_label(n):
        r=lookup.loc[n];kind='Playoffs' if r.season_type=='POST' else 'Regular season'
        return f"{r.visitor_team_abbr} at {r.home_team_abbr} · {kind}, week {int(r.week)}"
    game=st.selectbox('Matchup',g.game_id,format_func=game_label)
    plays=table('player_play.csv');eligible=plays[plays.game_id.eq(game)]
    names=table('players.csv').set_index('nfl_id').display_name.to_dict()
    if eligible.empty:st.info('No player performance records are available for this matchup.');return
    player=st.selectbox('Player',sorted(eligible.nfl_id.unique(),key=lambda n:str(names.get(n,''))),format_func=lambda n:names.get(n,'Player name unavailable'))
    if st.button('Explore this player',type='primary'):
        with st.spinner('Finding this player’s movement. The first load can take several minutes…'):
            st.session_state['tracking_selection']=(filename,int(game),int(player),frames(filename,int(game),int(player)))
    selection=st.session_state.get('tracking_selection')
    if selection is None or selection[:3]!=(filename,int(game),int(player)):return
    f=selection[3]
    if f.empty:st.info('No movement recordings are available for this player and game.');return
    play_ids=sorted(f.play_id.unique());play_names={}
    for i,pid in enumerate(play_ids,1):
        desc=eligible[(eligible.nfl_id.eq(player))&eligible.play_id.eq(pid)]
        text=str(desc.play_description.iloc[0]) if not desc.empty else ''
        play_names[pid]=f'Play {i}'+(f' · {text}' if text and text!='nan' else '')
    play=st.selectbox('Play',play_ids,format_func=lambda n:play_names[n]);q=f[f.play_id.eq(play)].sort_values('time')
    st.subheader(f'{names.get(player,"Selected player")} · {game_label(game)}')
    fig,ax=plt.subplots(figsize=(12,5));fig.patch.set_facecolor('#f5f7fb');ax.set_facecolor('#123e31')
    ax.set(xlim=(0,120),ylim=(0,53.33),xlabel='Length of field (yards)',ylabel='Width of field (yards)')
    for x in range(10,111,10):ax.axvline(x,color='white',alpha=.25)
    ax.axvspan(0,10,color='#102638');ax.axvspan(110,120,color='#102638')
    ax.plot(q.x,q.y,color='#65e1bc',linewidth=2.5);ax.scatter(q.x.iloc[0],q.y.iloc[0],c='white',label='Start',zorder=3)
    ax.scatter(q.x.iloc[-1],q.y.iloc[-1],c='#ffc857',label='End',zorder=3);ax.legend();ax.set_aspect('equal')
    st.pyplot(fig);plt.close(fig)
    st.caption('The path shows only the selected player. Offense may move in either direction.')
    with st.expander('Movement details'):
        detail_table(q)

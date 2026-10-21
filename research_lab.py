"""Exploratory player-level research; no predictive or causal claims."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from data_access import read_data, manifest
from presentation import LABELS, friendly

TESTS=['forty','ten_yd_split','vertical','broad_jump','three_cone','short_shuttle','bench_reps','combine_height','combine_weight','arm_length','hand_size','wing_span']
LOWER={'forty','ten_yd_split','three_cone','short_shuttle'}
OUTCOMES={
 'separation':('Average separation when thrown (yards)','separation_at_pass_forward','mean'),
 'yac':('Average yards after catch','yards_after_catch','mean'),
 'receiving':('Recorded receiving yards','rec_yards','sum'),
 'rushing':('Recorded rushing yards','rush_yards','sum'),
 'sacks':('Recorded sacks','sack','sum'),
 'getoff':('Average pass-rush get-off time (seconds)','player_get_off','mean'),
 'games':('Games with recorded plays',None,None),
 'plays':('Recorded player-play appearances',None,None),
}

def assemble(players,combine,plays,games):
    p=players.merge(combine.drop(columns='draft_year',errors='ignore'),on='nfl_id',how='left',validate='one_to_one')
    g=games[['game_id','season','season_type']].drop_duplicates('game_id')
    q=plays.merge(g,on='game_id',how='left',validate='many_to_one').merge(players[['nfl_id','draft_year']],on='nfl_id',validate='many_to_one')
    q=q[q.season.eq(q.draft_year)&q.season_type.eq('REG')].copy()
    # Do not count wiped-out plays in either outcome or observation denominator.
    if 'play_nullified_by_penalty' in q:q=q[q.play_nullified_by_penalty.eq('N')]
    summary=q.groupby('nfl_id').agg(plays=('play_id','size'),games=('game_id','nunique'))
    summary['plays_n']=summary.plays;summary['games_n']=summary.games
    for key,(_,col,method) in OUTCOMES.items():
        if col is None:continue
        valid=q[['nfl_id',col]].copy();valid[col]=pd.to_numeric(valid[col],errors='coerce')
        group=valid.groupby('nfl_id')[col]
        summary[key]=group.mean() if method=='mean' else group.sum(min_count=1)
        summary[key+'_n']=group.count()
    return p.merge(summary,on='nfl_id',how='left',validate='one_to_one')

@st.cache_data(max_entries=1)
def load():
    return assemble(*[read_data(n) for n in ['players.csv','combine_results.csv','player_play.csv','games.csv']])

def show():
    st.title('Research Lab')
    st.write('Compare athletic profiles and explore how combine testing relates to rookie NFL performance.')
    needed={'players.csv','combine_results.csv','player_play.csv','games.csv'}
    if not needed.issubset(manifest()):
        st.info('Player information, combine results, game schedule, and play performance are required. Check Data coverage.');return
    with st.spinner('Preparing rookie-season comparisons…'):data=load()
    a,b=st.columns(2)
    pos=a.selectbox('Position',sorted(data.nfl_position.dropna().unique()),index=list(sorted(data.nfl_position.dropna().unique())).index('WR'))
    years=b.multiselect('Draft classes',sorted(data.draft_year.unique()),default=sorted(data.draft_year.unique()))
    pool=data[data.nfl_position.eq(pos)&data.draft_year.isin(years)].copy()
    if pool.empty:st.info('Select at least one draft class.');return
    st.caption(f'{len(pool)} prospects · Same-position comparisons · Regular-season rookie year only')
    mode=st.radio('Research tool',['Player comparison','Combine vs. performance','Similar prospects'],horizontal=True)
    st.info('Exploratory research: relationships do not prove causation or predictive value. Players without recorded rookie plays have unknown outcomes here, not zero performance. Missing combine tests are not imputed.')
    names=pool.set_index('nfl_id').display_name.to_dict();ids=sorted(names,key=lambda n:names[n])
    if mode=='Player comparison':
        selected=st.multiselect('Compare players',ids,default=ids[:min(3,len(ids))],format_func=lambda n:names[n])
        measurements=st.multiselect('Measurements',TESTS,default=['forty','ten_yd_split','vertical','broad_jump'],format_func=lambda n:LABELS.get(n,n))
        if not selected or not measurements:return
        st.dataframe(friendly(pool[pool.nfl_id.isin(selected)][['display_name','draft_year']+measurements]).round(2),hide_index=True,use_container_width=True)
        rows=[]
        for metric in measurements:
            ranks=pool[metric].rank(pct=True,ascending=metric not in LOWER)*100
            for idx,row in pool[pool.nfl_id.isin(selected)].iterrows():
                rows.append({'Player':row.display_name,'Measurement':LABELS.get(metric,metric),'Position percentile':ranks.loc[idx],'Players tested':int(pool[metric].notna().sum())})
        st.subheader('Position percentiles')
        st.dataframe(pd.DataFrame(rows).round(1),hide_index=True,use_container_width=True)
        st.caption('Percentiles use recorded tests in the selected position and classes. Higher means a larger measurement; for timed drills, higher means faster. Size percentiles do not imply better performance. Ties use average ranks.')
    elif mode=='Combine vs. performance':
        col1,col2=st.columns(2)
        metric=col1.selectbox('Combine measurement',TESTS,format_func=lambda n:LABELS.get(n,n))
        outcome=col2.selectbox('Rookie outcome',list(OUTCOMES),format_func=lambda n:OUTCOMES[n][0])
        minimum=st.number_input('Minimum recorded observations for the selected outcome',min_value=1,max_value=2000,value=20 if outcome in {'separation','yac','getoff'} else 1)
        eligible=pool[pool[outcome+'_n'].ge(minimum)&pool[outcome].notna()]
        pairs=eligible.dropna(subset=[metric,outcome])
        c1,c2,c3=st.columns(3);c1.metric('Players compared',len(pairs));c2.metric('Missing selected combine test',int(eligible[metric].isna().sum()));c3.metric('Insufficient / missing NFL outcome',len(pool)-len(eligible))
        if len(pairs)<5 or pairs[metric].nunique()<2 or pairs[outcome].nunique()<2:
            st.warning('Not enough variation or at least five complete player records for this comparison. Try another measurement or outcome.');return
        pearson=pairs[metric].corr(pairs[outcome]);spearman=pairs[metric].rank().corr(pairs[outcome].rank())
        c1,c2=st.columns(2);c1.metric('Linear correlation',f'{pearson:+.2f}');c2.metric('Rank correlation',f'{spearman:+.2f}')
        fig,ax=plt.subplots(figsize=(10,5));ax.scatter(pairs[metric],pairs[outcome],color='#147d92',alpha=.75)
        ax.set(xlabel=LABELS.get(metric,metric),ylabel=OUTCOMES[outcome][0]);ax.spines[['top','right']].set_visible(False);fig.tight_layout();st.pyplot(fig);plt.close(fig)
        view=pairs[['display_name','draft_year',metric,outcome,outcome+'_n']].rename(columns={outcome:OUTCOMES[outcome][0],outcome+'_n':'Recorded observations'})
        st.dataframe(friendly(view).round(2),hide_index=True,use_container_width=True)
        st.caption('Each dot is one player; each player receives equal weight. Correlations range from −1 to +1. Timed tests run in the opposite direction: a smaller number is faster. Repeatedly trying combinations can reveal chance patterns; validate a chosen hypothesis on held-out players before claiming prediction.')
        st.caption('Recorded appearances and yardage totals reflect both opportunity and performance. Observation counts are non-missing records for the selected outcome, not automatically route, catch, or pass-rush counts. Confirm field applicability before interpreting another position.')
    else:
        player=st.selectbox('Find comparisons for',ids,format_func=lambda n:names[n])
        metrics=st.multiselect('Compare these measurements',TESTS,default=['forty','vertical','broad_jump','combine_height','combine_weight'],format_func=lambda n:LABELS.get(n,n))
        if not metrics:return
        target=pool[pool.nfl_id.eq(player)].iloc[0]
        missing=[m for m in metrics if pd.isna(target[m])]
        if missing:st.warning('Selected player is missing: '+', '.join(LABELS.get(m,m) for m in missing)+'. Remove those measurements to continue.');return
        candidates=pool.dropna(subset=metrics).copy()
        spread=pool[metrics].std(ddof=0);usable=[m for m in metrics if pd.notna(spread[m]) and spread[m]>0]
        if not usable:st.info('These measurements have no usable variation.');return
        if len(usable)<len(metrics):st.caption('Measurements without variation were excluded.')
        # A mixed-type player row is an object Series even after selecting
        # numeric fields. Convert explicitly before NumPy arithmetic.
        candidate_values=candidates[usable].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
        target_values=pd.to_numeric(target[usable],errors='raise').to_numpy(dtype=float)
        scales=spread[usable].to_numpy(dtype=float)
        candidates['Profile distance']=np.sqrt(np.mean(((candidate_values-target_values)/scales)**2,axis=1))
        nearest=candidates[candidates.nfl_id.ne(player)].nsmallest(10,'Profile distance')
        if nearest.empty:st.info('No other players have all selected measurements.');return
        outcome=st.selectbox('Show rookie outcome',list(OUTCOMES),format_func=lambda n:OUTCOMES[n][0])
        view=nearest[['display_name','draft_year','Profile distance']+usable+[outcome,outcome+'_n']].rename(columns={outcome:OUTCOMES[outcome][0],outcome+'_n':'Recorded observations'})
        st.dataframe(friendly(view).round(2),hide_index=True,use_container_width=True)
        st.caption(f'{len(candidates)-1} complete comparison candidates. Smaller distance means a more similar profile. Distances average squared differences after scaling each measurement by its position-group standard deviation. These are athletic comparisons, not NFL projections; no success label is used to select neighbors.')
    with st.expander('Research design and limits'):
        st.write('Rookie year is defined by game season matching draft year; playoffs and nullified plays are excluded. A player without recorded plays remains missing. No career totals are used as the rookie outcome. The sample includes only supplied 2023–2025 prospects. Injuries, team role, coaching, quarterback play, and selection into combine drills can affect results. No causal model or held-out validation has been run in this view. Combine movement features will be added after their measurement is validated.')

"""NFL career context from supplied career summaries and player-play records."""
import pandas as pd
import streamlit as st
from data_access import read_data, manifest
from presentation import detail_table

@st.cache_data(max_entries=1)
def sources():
    players=read_data('players.csv')
    games=read_data('games.csv')[['game_id','season','season_type']].drop_duplicates('game_id')
    plays=read_data('player_play.csv').merge(games,on='game_id',how='left',validate='many_to_one')
    careers=read_data('player_career_successes.csv') if 'player_career_successes.csv' in manifest() else None
    return players,plays,careers

def summarize(plays):
    records=[]
    for season,q in plays.groupby('season'):
        row={'Season':int(season),'Games with recorded plays':q.game_id.nunique(),'Recorded play appearances':len(q)}
        for col,label in [('rec_yards','Receiving yards'),('rush_yards','Rushing yards'),('sack','Sacks'),('tackle','Solo tackles')]:
            v=pd.to_numeric(q[col],errors='coerce')
            row[label]=v.sum(min_count=1)
        for col,label in [('separation_at_pass_forward','Separation when thrown (yards)'),('yards_after_catch','Yards after catch'),('player_get_off','Get-off time (seconds)')]:
            v=pd.to_numeric(q[col],errors='coerce')
            row[label]=v.mean();row[label+' — observations']=v.count()
        records.append(row)
    return pd.DataFrame(records)


def guide():
    with st.expander('How to interpret NFL success',expanded=False):
        st.markdown('''**Opportunity** describes chances to contribute: games active, starts, snaps, and recorded play appearances. More opportunity can reflect role, health, and coaching decisions.

**Performance** describes what happened during those chances: yards, separation, sacks, and movement timing. Totals mix opportunity and performance. Averages need their observation counts alongside them.

**Recognition** includes Pro Bowl and All-Pro selections. These are meaningful achievements, but are rare and reflect selection processes as well as performance.

Use all three perspectives. No single number establishes that one player is better. Compare similar positions and career stages, then investigate role, injuries, routes, coverage, and teammates.''')


def show():
    st.title('NFL Success')
    st.write('Follow a prospect from rookie year to later seasons, with playing opportunity and performance shown separately.')
    required={'players.csv','player_play.csv','games.csv'}
    if not required.issubset(manifest()):st.info('Player information, games, and NFL performance records are required.');return
    with st.spinner('Preparing NFL career context…'):players,plays,careers=sources()
    guide()
    col1,col2=st.columns(2)
    position=col1.selectbox('Position',sorted(players.nfl_position.dropna().unique()),index=list(sorted(players.nfl_position.dropna().unique())).index('WR'))
    pool=players[players.nfl_position.eq(position)];names=pool.set_index('nfl_id').display_name.to_dict()
    player=col2.selectbox('Player',sorted(names,key=lambda n:names[n]),format_func=lambda n:names[n])
    selected=pool[pool.nfl_id.eq(player)].iloc[0];rookie=int(selected.draft_year)
    st.subheader(names[player]);st.caption(f'{position} · {selected.college_name} · {rookie} draft class')
    q=plays[plays.nfl_id.eq(player)].copy()
    if 'play_nullified_by_penalty' in q:q=q[q.play_nullified_by_penalty.eq('N')]
    career_tab,season_tab,comparison_tab=st.tabs(['Career overview','Season by season','Compare players'])
    with career_tab:
        if careers is None:st.info('Career summaries are not available yet.')
        else:
            c=careers[careers.nfl_id.eq(player)]
            if c.empty:st.info('No career summary was supplied for this player.')
            else:
                r=c.iloc[0];a,b,d=st.columns(3)
                a.metric('Career games active',int(r.career_games_active));b.metric('Career starts',int(r.career_games_started))
                d.metric('First-team All-Pro selections',int(r.ap_all_pro_1st_team))
                st.markdown('#### Playing opportunity and recognition')
                detail_table(c)
        st.caption('Career summaries use the source file’s cumulative snapshot; its exact cutoff is not supplied. Offensive, defensive, and special teams snaps include regular season and playoffs. Games active count regular-season games. Do not divide those snaps by regular-season games and call it a regular-season snap rate.')
        st.markdown('#### What to investigate next')
        st.write('Did this player earn more opportunity over time? Did performance change as their role grew? Do athletic peers show similar outcomes, or does this player challenge the initial hypothesis?')
    with season_tab:
        stage=st.radio('Career stage',['All recorded seasons','Rookie season only'],horizontal=True)
        kind=st.radio('Games included',['Regular season','Playoffs'],horizontal=True)
        filtered=q[q.season_type.eq('REG' if kind=='Regular season' else 'POST')]
        if stage=='Rookie season only':filtered=filtered[filtered.season.eq(rookie)]
        summary=summarize(filtered)
        if summary.empty:st.info('No performance records match these filters. Missing records do not imply zero production.')
        else:
            summary.insert(1,'Career year',summary.Season-rookie+1)
            st.markdown('#### Playing opportunity')
            st.dataframe(summary[['Season','Career year','Games with recorded plays','Recorded play appearances']],hide_index=True,use_container_width=True)
            choices={
                'Receiving production':['Receiving yards','Separation when thrown (yards)','Yards after catch'],
                'Rushing production':['Rushing yards'],
                'Defensive production':['Sacks','Solo tackles','Get-off time (seconds)']}
            category=st.selectbox('Performance area',list(choices))
            metrics=choices[category];display=['Season','Career year']
            for m in metrics:
                display.append(m)
                if m+' — observations' in summary:display.append(m+' — observations')
            st.dataframe(summary[display].round(2),hide_index=True,use_container_width=True)
            available=[m for m in metrics if summary[m].notna().any()]
            if available:
                metric=st.selectbox('Trend to view',available)
                st.bar_chart(summary.set_index('Season')[[metric]])
            with st.expander('What these performance measures mean'):
                st.markdown('''- **Receiving and rushing yards:** sums from the supplied player-play records. They reflect opportunities as well as execution.
- **Separation when thrown:** average distance to the nearest defender at pass release, in yards. This can include routes where the player was not targeted. Route type, coverage, and timing matter.
- **Yards after catch:** average of non-missing supplied YAC values. Check observation counts and whether values are recorded only on catches before interpreting this as a per-catch rate.
- **Sacks:** sum of credited sacks; a shared sack may count as 0.5.
- **Solo tackles:** sum of credited solo tackles, excluding assists.
- **Get-off time:** average seconds from the snap until the defender moves forward across the line. Lower is faster; it does not establish pressure or sack success.
- **Observations:** non-missing records used for an average. They are not automatically counts of catches, routes, or pass-rush attempts.''')
        st.caption('Season summaries are calculated from the supplied player-play records, not verified against official career totals. Nullified plays are excluded. Zero, missing, and not applicable are different; avoid interpreting another position’s fields without checking applicability.')
    with comparison_tab:
        st.write('Compare the same position and draft class to keep time in the league consistent.')
        comparison_pool=players[players.nfl_position.eq(position)&players.draft_year.eq(rookie)]
        ids=comparison_pool.nfl_id.tolist();all_names=players.set_index('nfl_id').display_name.to_dict()
        all_players=st.checkbox('Select all players',key='career_all_players')
        selected_ids=ids if all_players else st.multiselect('Players to compare',sorted(ids,key=lambda n:all_names[n]),default=[player],format_func=lambda n:all_names[n])
        st.caption(f'{len(selected_ids)} players selected · {position} · {rookie} draft class')
        if careers is not None and selected_ids:
            joined=comparison_pool[comparison_pool.nfl_id.isin(selected_ids)][['nfl_id','display_name','draft_year','draft_round','draft_overall_pick']].merge(careers,on='nfl_id',how='left',validate='one_to_one')
            st.markdown('#### Same-class career summaries')
            detail_table(joined)
        st.markdown('#### Team discussion prompts')
        st.markdown('''1. State the hypothesis before trying more filters: “We expect ___ because ___.”
2. Separate opportunity from performance. Does the pattern remain among players with meaningful participation?
3. Inspect exceptions and missing tests. Are one or two players driving the result?
4. Check whether the pattern holds within a draft class, route, or coverage situation.
5. Reserve players or a draft class for validation before building a predictive claim.''')
        st.caption('Use this as a shared discussion guide. Notes are not saved or synchronized by this app.')

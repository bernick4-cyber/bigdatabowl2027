"""Rookie defensive-line summaries and chronological prediction evaluation."""
import numpy as np
import pandas as pd
from data_access import read_data

POSITIONS = {'DE': 'Edge (DE)', 'DT': 'Interior (DT / NT)', 'NT': 'Interior (DT / NT)'}
FEATURES = ['ten_yd_split', 'broad_jump', 'vertical', 'combine_weight', 'arm_length', 'wing_span']
BASELINE = ['ngs_athleticism_score']
DROPBACK_SCOPE = 'Identified dropbacks (pass-rush proxy)'
ALL_SCOPE = 'All recorded plays (reference)'


def identified_dropbacks(plays):
    """Explicit pass results and sacks; R is included only for described scrambles.

    This is a play-context proxy, not a verified individual rush assignment.
    Missing/unknown codes are excluded, not assumed to be runs or dropbacks.
    """
    code = plays.pass_result.astype('string').str.strip().str.upper()
    scramble = plays.play_description.astype('string').str.contains(r'\bscrambl\w*\b', case=False, na=False)
    return (code.isin(['C', 'I', 'IN', 'S']) | (code.eq('R') & scramble)).fillna(False)


def summarize_rookies(plays, players, games, scope=DROPBACK_SCOPE):
    """Count only supplied, non-nullified regular-season rookie observations."""
    q = plays.merge(players[['nfl_id', 'draft_year']], on='nfl_id', validate='many_to_one')
    q = q.merge(games[['game_id', 'season', 'season_type']], on='game_id', validate='many_to_one')
    q = q[q.season.eq(q.draft_year) & q.season_type.eq('REG')
          & q.play_nullified_by_penalty.eq('N')].copy()
    q['player_get_off'] = pd.to_numeric(q.player_get_off, errors='coerce')
    q['sack'] = pd.to_numeric(q.sack, errors='coerce')
    audit = q.groupby('nfl_id').agg(all_getoff=('player_get_off', 'mean'),
        all_getoff_n=('player_get_off', 'count'), all_recorded_plays=('play_id', 'size'))
    mask = identified_dropbacks(q)
    passing = q[mask].groupby('nfl_id').agg(dropback_getoff=('player_get_off', 'mean'),
        dropback_getoff_n=('player_get_off', 'count'), identified_dropback_plays=('play_id', 'size'))
    audit = audit.join(passing)
    for col in ['dropback_getoff_n', 'identified_dropback_plays']:
        audit[col] = audit[col].fillna(0).astype(int)
    audit['excluded_context_getoff_n'] = audit.all_getoff_n - audit.dropback_getoff_n
    if scope == DROPBACK_SCOPE:
        q = q[mask]
    elif scope != ALL_SCOPE:
        raise ValueError('Unknown play sample.')
    summary = q.groupby('nfl_id').agg(
        getoff=('player_get_off', 'mean'), getoff_n=('player_get_off', 'count'),
        recorded_sacks=('sack', lambda x: x.sum(min_count=1)),
        sack_n=('sack', 'count'), recorded_plays=('play_id', 'size'),
        recorded_games=('game_id', 'nunique'))
    result = audit.join(summary)
    for col in ['getoff_n', 'sack_n', 'recorded_plays', 'recorded_games']:
        result[col] = result[col].fillna(0).astype(int)
    result['sample_scope'] = scope
    return result


def load_defensive_line(scope=DROPBACK_SCOPE):
    players = read_data('players.csv')
    players = players[players.nfl_position.isin(POSITIONS)].copy()
    combine = read_data('combine_results.csv').drop(columns='draft_year', errors='ignore')
    games = read_data('games.csv')
    pieces = []
    columns = ['nfl_id', 'game_id', 'play_id', 'play_nullified_by_penalty', 'player_get_off', 'sack', 'pass_result', 'play_description']
    with read_data('player_play.csv', chunksize=100000) as reader:
        for chunk in reader:
            pieces.append(chunk.loc[chunk.nfl_id.isin(players.nfl_id), columns])
    plays = pd.concat(pieces, ignore_index=True)
    if plays.duplicated(['game_id', 'play_id', 'nfl_id']).any():
        raise ValueError('Duplicate player-play rows must be resolved before analysis.')
    summary = summarize_rookies(plays, players, games, scope)
    result = players.merge(combine, on='nfl_id', how='left', validate='one_to_one')
    result = result.merge(summary, on='nfl_id', how='left', validate='one_to_one')
    result['line_group'] = result.nfl_position.map(POSITIONS)
    result['sample_scope'] = scope
    return result


def fit_predict(train, test, columns):
    """OLS, with scaling and position coding fitted using training players only."""
    numeric = list(columns)
    a = train[numeric].to_numpy(float)
    b = test[numeric].to_numpy(float)
    center = a.mean(axis=0)
    scale = a.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    a, b = (a - center) / scale, (b - center) / scale
    groups = sorted(train.line_group.unique())
    if not set(test.line_group).issubset(groups):
        raise ValueError('A test position group is absent from the training sample.')
    for group in groups[1:]:
        a = np.column_stack([a, train.line_group.eq(group).astype(float)])
        b = np.column_stack([b, test.line_group.eq(group).astype(float)])
    a = np.column_stack([np.ones(len(train)), a])
    b = np.column_stack([np.ones(len(test)), b])
    if len(train) < a.shape[1] + 5:
        raise ValueError('Too few training players for these variables; reduce measurements or broaden the group.')
    if np.linalg.matrix_rank(a) < a.shape[1]:
        raise ValueError('Selected measurements have no variation or repeat information; remove a measurement.')
    return b @ np.linalg.lstsq(a, train.getoff.to_numpy(float), rcond=None)[0]


def evaluate_holdout(data, features, minimum=20, test_year=2025):
    """Use a common complete sample; never fit on the held-out class."""
    columns = list(dict.fromkeys(BASELINE + list(features)))
    eligible = data[data.getoff_n.ge(minimum) & data.getoff.notna()]
    complete = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=columns + ['getoff'])
    train = complete[complete.draft_year.lt(test_year)].copy()
    test = complete[complete.draft_year.eq(test_year)].copy()
    if len(test) < 5:
        raise ValueError(f'Only {len(test)} complete test-class players remain; at least five are required. This group cannot support the current comparison. Keep the observation threshold consistent across reported results.')
    if train.empty:
        raise ValueError('No earlier draft classes are available for training.')
    predictions = test[['nfl_id', 'display_name', 'nfl_position', 'draft_year', 'getoff', 'getoff_n']].copy()
    predictions['sample_scope'] = test['sample_scope'] if 'sample_scope' in test else 'Unspecified'
    predictions['added_measurements'] = ', '.join(features) if features else 'None'
    predictions['minimum_getoff_observations'] = minimum
    configurations = [('Training average', None), ('Athleticism + position group', BASELINE)]
    if features:
        configurations.append(('Athleticism + position group + measurements', columns))
    scores = []
    for name, inputs in configurations:
        predicted = np.repeat(train.getoff.mean(), len(test)) if inputs is None else fit_predict(train, test, inputs)
        error = test.getoff.to_numpy(float) - predicted
        scores.append({'Model': name, 'MAE (seconds)': np.abs(error).mean(),
                       'RMSE (seconds)': np.sqrt(np.mean(error ** 2))})
        predictions[name] = predicted
    coverage = {'Eligible': len(eligible), 'Missing selected inputs': len(eligible) - len(complete),
                'Training players': len(train), 'Test players': len(test)}
    return pd.DataFrame(scores), predictions, coverage

import unittest
import numpy as np
import pandas as pd
from defensive_line_data import summarize_rookies, evaluate_holdout, identified_dropbacks, ALL_SCOPE


class DefensiveLineTests(unittest.TestCase):
    def test_rookie_regular_season_and_penalties(self):
        players = pd.DataFrame({'nfl_id': [1], 'draft_year': [2023]})
        games = pd.DataFrame({'game_id': [1, 2, 3], 'season': [2023, 2024, 2023], 'season_type': ['REG', 'REG', 'POST']})
        plays = pd.DataFrame({'nfl_id': [1]*5, 'game_id': [1, 1, 1, 2, 3],
                              'play_id': [1, 2, 3, 4, 5], 'play_nullified_by_penalty': ['N', 'N', 'Y', 'N', 'N'],
                              'player_get_off': [1., np.nan, 9., 9., 9.], 'sack': [0., np.nan, 1., 1., 1.],
                              'pass_result': ['C']*5, 'play_description': ['pass']*5})
        row = summarize_rookies(plays, players, games, ALL_SCOPE).loc[1]
        self.assertEqual(row.getoff, 1.)
        self.assertEqual(row.getoff_n, 1)
        self.assertEqual(row.recorded_plays, 2)
        self.assertEqual(row.recorded_sacks, 0.)

    def sample(self):
        n = 30
        return pd.DataFrame({'nfl_id': range(n), 'display_name': [str(i) for i in range(n)],
            'nfl_position': ['DT']*n, 'line_group': ['Interior']*n,
            'draft_year': [2023]*10 + [2024]*10 + [2025]*10,
            'getoff_n': [30]*n, 'getoff': np.linspace(.7, 1.3, n),
            'ngs_athleticism_score': np.sin(np.arange(n)) + 70,
            'ten_yd_split': np.cos(np.arange(n)) + 2})

    def test_test_outcomes_do_not_affect_predictions(self):
        d = self.sample()
        _, before, _ = evaluate_holdout(d, ['ten_yd_split'])
        d.loc[d.draft_year.eq(2025), 'getoff'] += 10
        _, after, _ = evaluate_holdout(d, ['ten_yd_split'])
        columns = ['Training average', 'Athleticism + position group', 'Athleticism + position group + measurements']
        np.testing.assert_allclose(before[columns].to_numpy(float), after[columns].to_numpy(float))

    def test_dropback_rule_handles_sacks_scrambles_and_unknowns(self):
        plays = pd.DataFrame({'pass_result': ['C','I','IN','S','R','R',None,'X','R'],
            'play_description': ['pass','pass','pass','sacked','QB scrambles left','QB up middle','run','pass',None]})
        self.assertEqual(identified_dropbacks(plays).tolist(), [True]*5 + [False]*4)

    def test_filter_changes_denominator_without_losing_audit(self):
        players = pd.DataFrame({'nfl_id':[1], 'draft_year':[2025]})
        games = pd.DataFrame({'game_id':[1], 'season':[2025], 'season_type':['REG']})
        plays = pd.DataFrame({'nfl_id':[1]*4, 'game_id':[1]*4, 'play_id':[1,2,3,4],
            'play_nullified_by_penalty':['N']*4, 'player_get_off':[1.,3.,None,2.],
            'sack':[0.,0.,0.,1.], 'pass_result':['C',None,'I','S'], 'play_description':['pass','run','incomplete','sacked']})
        row = summarize_rookies(plays,players,games).loc[1]
        self.assertEqual(row.getoff,1.5)
        self.assertEqual(row.getoff_n,2)
        self.assertEqual(row.recorded_plays,3)
        self.assertEqual(row.all_getoff_n,3)
        self.assertEqual(row.excluded_context_getoff_n,1)
        self.assertEqual(row.recorded_sacks,1.)
        reference = summarize_rookies(plays,players,games,ALL_SCOPE).loc[1]
        self.assertEqual(reference.getoff,2.)
        self.assertEqual(reference.getoff_n,3)

    def test_no_dropbacks_retains_player_audit(self):
        players = pd.DataFrame({'nfl_id':[1], 'draft_year':[2025]})
        games = pd.DataFrame({'game_id':[1], 'season':[2025], 'season_type':['REG']})
        plays = pd.DataFrame({'nfl_id':[1], 'game_id':[1], 'play_id':[1],
            'play_nullified_by_penalty':['N'], 'player_get_off':[1.], 'sack':[0.],
            'pass_result':[None], 'play_description':['run']})
        row = summarize_rookies(plays,players,games).loc[1]
        self.assertEqual(row.getoff_n,0)
        self.assertTrue(pd.isna(row.getoff))
        self.assertEqual(row.all_getoff_n,1)

    def test_common_sample_and_chronological_split(self):
        d = self.sample()
        d.loc[0, 'ten_yd_split'] = np.nan
        d.loc[20, 'ngs_athleticism_score'] = np.nan
        scores, predictions, coverage = evaluate_holdout(d, ['ten_yd_split'])
        self.assertEqual(coverage['Training players'], 19)
        self.assertEqual(coverage['Test players'], 9)
        self.assertEqual(coverage['Missing selected inputs'], 2)
        self.assertTrue(predictions.draft_year.eq(2025).all())
        self.assertEqual(len(scores), 3)

    def test_small_test_sample_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'five'):
            evaluate_holdout(self.sample().iloc[:24], ['ten_yd_split'])


if __name__ == '__main__':
    unittest.main()

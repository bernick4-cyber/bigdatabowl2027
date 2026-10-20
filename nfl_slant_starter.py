"""NFL Big Data Bowl: exploratory combine slant inspection.

Install once: python -m pip install pandas numpy matplotlib
Put this script beside players.csv and combine_tracking.csv (or
combine_tracking (1).csv), then run: python nfl_slant_starter.py
Alternatively: python nfl_slant_starter.py --data-dir upload
Outputs are written to slant_outputs/ by default.

Turn detection is an EXPLORATORY heuristic, not a validated football label.
Inspect the marked plots before interpreting the exported measurements.
This script does not fit a model or establish an NFL performance relationship.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inspect_attempt(g):
    g = g.sort_values('time').copy()
    g['time'] = pd.to_datetime(g['time'], errors='raise')
    seconds = (g.time - g.time.iloc[0]).dt.total_seconds().to_numpy()
    if len(g) < 9 or np.any(np.diff(seconds) <= 0):
        return g, None, None
    # Circular unwrapping avoids a false turn when headings cross 0/360.
    heading = np.unwrap(np.deg2rad(g.dir.to_numpy()))
    smooth = pd.Series(heading).rolling(3, center=True, min_periods=1).mean().to_numpy()
    rate = np.abs(np.gradient(smooth, seconds))
    # Exclude stationary starts and leave 0.3 seconds on either side.
    eligible = (g.s.to_numpy() >= 3) & (seconds >= .3) & (seconds <= seconds[-1] - .3)
    if not eligible.any():
        return g, None, None
    i = int(np.argmax(np.where(eligible, rate, -np.inf)))
    before = (seconds >= seconds[i] - .3) & (seconds < seconds[i])
    after = (seconds > seconds[i]) & (seconds <= seconds[i] + .3)
    entry = float(g.s.to_numpy()[before].mean())
    exit_speed = float(g.s.to_numpy()[after].mean())
    metrics = {
        'candidate_turn_seconds': seconds[i],
        'candidate_turn_x': g.x.iloc[i], 'candidate_turn_y': g.y.iloc[i],
        'entry_speed_yd_s': entry, 'exit_speed_yd_s': exit_speed,
        'exit_entry_speed_ratio': exit_speed / entry if entry > 0 else np.nan,
        'peak_heading_rate_deg_s': np.rad2deg(rate[i]),
        'frames': len(g), 'duration_seconds': seconds[-1],
        'method': 'peak_smoothed_heading_rate_while_moving_unvalidated',
    }
    return g, i, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path('slant_outputs'))
    args = parser.parse_args()
    tracking = args.data_dir / 'combine_tracking.csv'
    if not tracking.exists():
        tracking = args.data_dir / 'combine_tracking (1).csv'
    players = pd.read_csv(args.data_dir / 'players.csv')
    frames = pd.read_csv(tracking)
    names = players.set_index('nfl_id').display_name
    wr_ids = players.loc[players.nfl_position.eq('WR'), 'nfl_id']
    selected = frames[frames.nfl_id.isin(wr_ids) & frames.entity_type.eq('PLAYER')
                      & frames.drill_name.eq('SLANT_ROUTE_LEFT')].copy()
    if selected.empty:
        raise ValueError('No WR SLANT_ROUTE_LEFT player frames found.')
    if selected.duplicated(['event_id', 'nfl_id', 'time']).any():
        raise ValueError('Duplicate frames found; resolve before measuring.')
    if selected[['x', 'y', 's', 'dir', 'time']].isna().any().any():
        raise ValueError('Missing tracking values found; resolve before measuring.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    for (event, player), group in selected.groupby(['event_id', 'nfl_id'], sort=True):
        g, i, metrics = inspect_attempt(group)
        if metrics is None:
            print('Skipped unusable attempt:', event, player)
            continue
        rows.append(dict(event_id=event, nfl_id=int(player), display_name=names.loc[player],
                         draft_year=int(g.draft_year.iloc[0]), **metrics))
        if len(examples) < 6:
            examples.append((g, i, names.loc[player]))
    if not rows:
        raise ValueError('No attempts passed the measurement checks.')
    pd.DataFrame(rows).to_csv(args.output_dir / 'slant_attempt_metrics.csv', index=False)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, (g, i, name) in zip(axs.flat, examples):
        ax.set_visible(True)
        x, y = g.x - g.x.iloc[0], g.y - g.y.iloc[0]
        ax.plot(x, y, '-o', markersize=2)
        ax.scatter(x.iloc[i], y.iloc[i], color='red', s=45, label='Candidate turn')
        ax.set(title=name, xlabel='x displacement (yards)', ylabel='y displacement (yards)')
        ax.set_aspect('equal'); ax.legend(fontsize=8)
    fig.suptitle('Combine slants: inspect candidate turns before using metrics')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'slant_route_checks.png', dpi=160)
    plt.close(fig)
    print(f'Processed {len(rows)} attempts for {pd.DataFrame(rows).nfl_id.nunique()} receivers.')
    print(f'Outputs: {args.output_dir.resolve()}')
    print('These are exploratory measurements. Inspect the red markers before analysis.')


if __name__ == '__main__':
    main()

"""Read split gzip files without joining a season in RAM."""
import io, gzip, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent / 'data'
class Parts(io.RawIOBase):
    def __init__(self, paths):
        self.paths=iter(paths); self.current=None
    def readable(self): return True
    def readinto(self, buffer):
        while True:
            if self.current is None:
                try: self.current=open(next(self.paths),'rb')
                except StopIteration: return 0
            count=self.current.readinto(buffer)
            if count: return count
            self.current.close(); self.current=None
    def close(self):
        if self.current is not None: self.current.close()
        super().close()
def manifest(): return json.loads((ROOT/'manifest.json').read_text())
def read_data(name, chunksize=None):
    parts=manifest()[name]['parts']
    # pandas owns the resulting stream until its reader is closed.
    stream=gzip.GzipFile(fileobj=io.BufferedReader(Parts([ROOT/p for p in parts])))
    if chunksize:
        return pd.read_csv(stream,chunksize=chunksize)
    try: return pd.read_csv(stream)
    finally: stream.close()
def select_tracking(name, game_id, nfl_id):
    matches=[]
    with read_data(name,100000) as reader:
        for chunk in reader:
            match=chunk[chunk.game_id.eq(game_id)&chunk.nfl_id.eq(nfl_id)]
            if not match.empty: matches.append(match)
    return pd.concat(matches,ignore_index=True) if matches else pd.DataFrame()

import asyncio
from pathlib import Path

import pandas as pd

TICK_SECONDS = 1.0
CSV_PATH = Path(__file__).resolve().parents[1] / "PW_0.5h" / "test_failure.csv"
# CSV_PATH = Path(__file__).resolve().parents[1] / "contamination_benchmarks" / "test" / "modified_5pct_rep01.csv"


class ReplayCursor:
    """Walk a saved sensor file one row at a time.

    ``index`` is the next row to publish. Pause holds that position, play
    continues, and restart returns to the first row and starts playing.
    """

    def __init__(self, path: Path = CSV_PATH, tick_seconds: float = TICK_SECONDS) -> None:
        frame = pd.read_csv(path)
        self.rows = frame.to_dict(orient="records")
        self.index = 0
        self.playing = True
        self.tick_seconds = tick_seconds
        self._control = asyncio.Event()

    def next_row(self) -> dict:
        row = self.rows[self.index]
        payload = {
            "index": self.index,
            "total": len(self.rows),
            "record": row,
        }
        self.index = (self.index + 1) % len(self.rows)
        return payload

    def status(self) -> dict:
        return {
            "playing": self.playing,
            "next_index": self.index,
            "total": len(self.rows),
        }

    def pause(self) -> dict:
        self.playing = False
        self._control.set()
        return self.status()

    def play(self) -> dict:
        self.playing = True
        self._control.set()
        return self.status()

    def restart(self) -> dict:
        self.index = 0
        self.playing = True
        self._control.set()
        return self.status()

    async def run(self, publish) -> None:
        while True:
            if self.playing:
                await publish(self.next_row())
            if self.playing:
                try:
                    await asyncio.wait_for(self._control.wait(), self.tick_seconds)
                except TimeoutError:
                    pass
            else:
                await self._control.wait()
            self._control.clear()

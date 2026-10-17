import asyncio
from pathlib import Path

import pandas as pd

TICK_SECONDS = 1.0
CSV_PATH = Path(__file__).resolve().parents[1] / "PW_0.5h" / "test.csv"


class ReplayCursor:
    def __init__(self, path: Path = CSV_PATH) -> None:
        frame = pd.read_csv(path)
        self.rows = frame.to_dict(orient="records")
        self.index = 0

    def next_row(self) -> dict:
        row = self.rows[self.index]
        payload = {
            "index": self.index,
            "total": len(self.rows),
            "record": row,
        }
        self.index = (self.index + 1) % len(self.rows)
        return payload

    async def run(self, publish) -> None:
        while True:
            await publish(self.next_row())
            await asyncio.sleep(TICK_SECONDS)

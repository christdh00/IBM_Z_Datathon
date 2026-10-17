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
            "timestamp": str(row["timestamp"]),
            "cms_air_pressure": float(row["CMS air pressure"]),
            "oxygen_concentration": float(row["Concentrazione ossigeno base"]),
            "nitrogen_pressure": float(row["Nitrogen pressure"]),
            "oxygen_over_threshold": float(row["Oxygen over threshold"]),
            "pw_0_5h": int(row["PW_0.5h"]),
        }
        self.index = (self.index + 1) % len(self.rows)
        return payload

    async def run(self, publish) -> None:
        while True:
            await publish(self.next_row())
            await asyncio.sleep(TICK_SECONDS)

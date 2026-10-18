import asyncio
from pathlib import Path

import pandas as pd
import pytest

from app.payload import with_replay
from app.replay import ReplayCursor


def _cursor(tmp_path: Path, rows: int = 3, tick_seconds: float = 30) -> ReplayCursor:
    path = tmp_path / "replay.csv"
    pd.DataFrame(
        {
            "timestamp": [f"2023-09-22 00:0{index}:00" for index in range(rows)],
            "CMS air pressure": [0.1] * rows,
        }
    ).to_csv(path, index=False)
    return ReplayCursor(path, tick_seconds=tick_seconds)


def test_next_row_wraps_to_the_start(tmp_path):
    cursor = _cursor(tmp_path, rows=2)

    assert cursor.next_row()["index"] == 0
    assert cursor.next_row()["index"] == 1
    wrapped = cursor.next_row()

    assert wrapped["index"] == 0
    assert wrapped["total"] == 2
    assert wrapped["record"]["CMS air pressure"] == 0.1


def test_pause_play_and_restart_report_cursor_state(tmp_path):
    cursor = _cursor(tmp_path)

    assert cursor.status() == {"playing": True, "next_index": 0, "total": 3}
    assert cursor.pause()["playing"] is False
    cursor.next_row()
    assert cursor.play() == {"playing": True, "next_index": 1, "total": 3}
    assert cursor.restart() == {"playing": True, "next_index": 0, "total": 3}


def test_run_pauses_plays_and_restarts(tmp_path):
    cursor = _cursor(tmp_path, rows=5, tick_seconds=0.05)
    seen = []

    async def scenario():
        published = asyncio.Event()

        async def publish(row):
            seen.append(row["index"])
            published.set()

        task = asyncio.create_task(cursor.run(publish))
        await asyncio.wait_for(published.wait(), 1)
        assert seen == [0]
        cursor.pause()
        await asyncio.sleep(0.25)
        assert seen == [0]
        assert cursor.playing is False

        published.clear()
        cursor.play()
        await asyncio.wait_for(published.wait(), 1)
        assert seen == [0, 1]

        published.clear()
        cursor.restart()
        await asyncio.wait_for(published.wait(), 1)
        assert seen == [0, 1, 0]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_with_replay_marks_whether_the_cursor_is_advancing():
    payload = {"index": 4}

    assert with_replay(payload, False) is payload
    assert payload["replay_playing"] is False
    assert payload["index"] == 4


def test_replay_http_controls_and_frontend(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    async def idle(_publish):
        await asyncio.Event().wait()

    monkeypatch.setattr(main.cursor, "run", idle)
    monkeypatch.setattr(main.cursor, "playing", main.cursor.playing)
    monkeypatch.setattr(main.cursor, "index", main.cursor.index)
    main.cursor.pause()

    with TestClient(main.app) as client:
        assert client.get("/api/replay").json()["playing"] is False
        assert client.get("/api/snapshot").json() is None
        assert client.post("/api/replay/play").json()["playing"] is True
        assert client.post("/api/replay/pause").json()["playing"] is False
        restarted = client.post("/api/replay/restart").json()
        assert restarted == {"playing": True, "next_index": 0, "total": restarted["total"]}
        page = client.get("/")
        assert page.status_code == 200
        assert 'id="pauseButton"' in page.text
        assert 'id="playButton"' in page.text
        assert 'id="restartButton"' in page.text
        assert client.get("/dashboard.js").status_code == 200
        assert client.get("/style.css").status_code == 200

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from app.inference import (
    FailureModel,
    IsolationForestModel,
    RandomForestModel,
    SurvivalModel,
)
from app.payload import combine, with_replay
from app.readings import ReadingWindow
from app.replay import ReplayCursor

cursor = ReplayCursor()
failure_model = FailureModel()
forest_model = RandomForestModel()
survival_model = SurvivalModel()
isolation_model = IsolationForestModel()
window = ReadingWindow(
    failure_model.window_minutes,
    buffer_minutes=max(
        failure_model.window_minutes,
        survival_model.window_minutes,
        isolation_model.window_minutes,
    ),
)
latest: dict | None = None
subscribers: set[asyncio.Queue] = set()
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "trustguard_frontend"


def broadcast(payload: dict) -> None:
    for queue in list(subscribers):
        queue.put_nowait(payload)


def notify_replay() -> None:
    if latest is None:
        return
    broadcast(with_replay(latest, cursor.playing))


async def publish(row: dict) -> None:
    global latest
    reading = window.observe(row)
    failure = None
    forest = None
    if len(reading.rows) >= failure_model.window_minutes:
        window_rows = reading.rows[-failure_model.window_minutes :]
        failure = failure_model.predict(window_rows)
        forest = forest_model.predict(window_rows)
    survival = None
    if len(reading.rows) >= survival_model.window_minutes:
        survival = survival_model.predict(reading.rows)
    isolation = None
    if len(reading.rows) >= isolation_model.window_minutes:
        isolation = isolation_model.predict(
            reading.rows[-isolation_model.window_minutes :]
        )
    latest = with_replay(
        combine(
            reading,
            failure,
            survival,
            survival_model.window_minutes,
            forest,
            isolation,
            isolation_model.window_minutes,
        ),
        cursor.playing,
    )
    broadcast(latest)


def close_streams() -> None:
    """End open replay streams so a reload is not stuck on the event source."""
    for queue in list(subscribers):
        queue.put_nowait(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cursor.run(publish))
    try:
        yield
    finally:
        close_streams()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)


def _frontend(name: str, media_type: str | None = None) -> FileResponse:
    return FileResponse(
        FRONTEND_DIR / name,
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/")
async def page() -> FileResponse:
    return _frontend("index.html")


@app.get("/style.css")
async def styles() -> FileResponse:
    return _frontend("style.css", "text/css")


@app.get("/dashboard.js")
async def dashboard_script() -> FileResponse:
    return _frontend("dashboard.js", "text/javascript")


@app.get("/api/snapshot")
async def snapshot():
    return latest


@app.get("/api/replay")
async def replay_status():
    return cursor.status()


@app.post("/api/replay/pause")
async def pause_replay():
    status = cursor.pause()
    notify_replay()
    return status


@app.post("/api/replay/play")
async def play_replay():
    status = cursor.play()
    notify_replay()
    return status


@app.post("/api/replay/restart")
async def restart_replay():
    return cursor.restart()


@app.get("/api/stream")
async def stream():
    queue: asyncio.Queue = asyncio.Queue()
    subscribers.add(queue)

    async def events():
        try:
            while True:
                row = await queue.get()
                if row is None:
                    break
                yield f"data: {json.dumps(row)}\n\n"
        finally:
            subscribers.discard(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

from fastapi import HTTPException
from pydantic import BaseModel
from .granite import generate_maintenance_advice
import asyncio

class AdviceRequest(BaseModel):
    predictions: dict

@app.post("/api/advice")
async def generate_advice(request: AdviceRequest):
    try:
        advice = await asyncio.to_thread(
            generate_maintenance_advice,
            request.predictions
        )
        return {"advice": advice}

    except Exception:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=502,
            detail="IBM Granite is unavailable"
        )
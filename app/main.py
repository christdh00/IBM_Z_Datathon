import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from app.inference import FailureModel
from app.payload import combine
from app.readings import ReadingWindow
from app.replay import ReplayCursor

cursor = ReplayCursor()
model = FailureModel()
window = ReadingWindow(model.window_minutes)
latest: dict | None = None
subscribers: set[asyncio.Queue] = set()

PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Replay</title>
</head>
<body>
  <p id="status">waiting for the first row</p>
  <p id="model"></p>
  <pre id="row"></pre>
  <script>
    const status = document.getElementById("status");
    const model = document.getElementById("model");
    const row = document.getElementById("row");
    function show(data) {
      status.textContent = (data.index + 1) + " / " + data.total;
      if (data.status === "too_small_window") {
        model.textContent = "too small window (" + data.window_rows + " / " + data.window_required + " minutes)";
      } else {
        const call = data.failure_predicted ? "failure" : "no failure";
        const probability = data.failure_probability >= 0.0001
          ? data.failure_probability.toFixed(4)
          : data.failure_probability.toExponential(2);
        model.textContent = "failure probability " + probability + " (" + call + ")";
      }
      row.textContent = JSON.stringify(data, null, 2);
    }
    fetch("/api/snapshot").then(r => r.json()).then(data => {
      if (data) show(data);
    });
    const source = new EventSource("/api/stream");
    source.onmessage = (event) => show(JSON.parse(event.data));
  </script>
</body>
</html>"""


async def publish(row: dict) -> None:
    global latest
    reading = window.observe(row)
    prediction = model.predict(reading.rows) if reading.rows is not None else None
    latest = combine(reading, prediction)
    for queue in list(subscribers):
        queue.put_nowait(latest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cursor.run(publish))
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def page() -> str:
    return PAGE


@app.get("/api/snapshot")
async def snapshot():
    return latest


@app.get("/api/stream")
async def stream():
    queue: asyncio.Queue = asyncio.Queue()
    subscribers.add(queue)

    async def events():
        try:
            while True:
                row = await queue.get()
                yield f"data: {json.dumps(row)}\n\n"
        finally:
            subscribers.discard(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

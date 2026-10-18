import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from app.inference import FailureModel, RandomForestModel, SurvivalModel
from app.payload import combine
from app.readings import ReadingWindow
from app.replay import ReplayCursor

cursor = ReplayCursor()
failure_model = FailureModel()
forest_model = RandomForestModel()
survival_model = SurvivalModel()
window = ReadingWindow(max(failure_model.window_minutes, survival_model.window_minutes))
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
  <p id="forest"></p>
  <p id="survival"></p>
  <pre id="row"></pre>
  <script>
    const status = document.getElementById("status");
    const model = document.getElementById("model");
    const forest = document.getElementById("forest");
    const survival = document.getElementById("survival");
    const row = document.getElementById("row");
    function formatProbability(value) {
      return value >= 0.0001 ? value.toFixed(4) : value.toExponential(2);
    }
    function show(data) {
      status.textContent = (data.index + 1) + " / " + data.total;
      if (data.status === "too_small_window") {
        model.textContent = "too small window (" + data.window_rows + " / " + data.window_required + " minutes)";
        forest.textContent = "random forest: too small window (" + data.window_rows + " / " + data.window_required + " minutes)";
      } else {
        const call = data.failure_predicted ? "failure" : "no failure";
        model.textContent = "xgboost failure probability " + formatProbability(data.failure_probability) + " (" + call + ")";
        const forestCall = data.forest_failure_predicted ? "failure" : "no failure";
        forest.textContent = "random forest probability " + formatProbability(data.forest_failure_probability) + " (" + forestCall + ")";
      }
      if (data.survival_status === "more_data_required") {
        survival.textContent = "more data required (" + data.window_rows + " / " + data.survival_window_required + " minutes)";
      } else {
        survival.textContent = data.survival_model
          + " — 30 min " + data.failure_probability_30_min.toFixed(4)
          + ", 60 min " + data.failure_probability_60_min.toFixed(4)
          + ", 120 min " + data.failure_probability_120_min.toFixed(4)
          + ", median " + data.median_time_to_failure_min.toFixed(4) + " min";
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
    failure = None
    forest = None
    if len(reading.rows) >= failure_model.window_minutes:
        window_rows = reading.rows[-failure_model.window_minutes :]
        failure = failure_model.predict(window_rows)
        forest = forest_model.predict(window_rows)
    survival = None
    if len(reading.rows) >= survival_model.window_minutes:
        survival = survival_model.predict(reading.rows)
    latest = combine(
        reading,
        failure,
        survival,
        survival_model.window_minutes,
        forest,
    )
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

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from app.replay import ReplayCursor

cursor = ReplayCursor()
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
  <pre id="row"></pre>
  <script>
    const status = document.getElementById("status");
    const row = document.getElementById("row");
    function show(data) {
      status.textContent = (data.index + 1) + " / " + data.total;
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
    latest = row
    for queue in list(subscribers):
        queue.put_nowait(row)


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

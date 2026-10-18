# TrustGuard frontend

The dashboard is served by the FastAPI app in `app/`. It listens to `/api/stream` and renders each payload: sensor readings, XGBoost, random forest, isolation forest, and survival scores.

## Run

From the repository root:

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Pause, Play, and Restart control the replay cursor. The page also shows the latest payload JSON.

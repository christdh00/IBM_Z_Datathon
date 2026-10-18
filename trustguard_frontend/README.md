# TrustGuard frontend (HTML/CSS/JavaScript)

This is a frontend-only prototype. There is no Python, Flask, or backend requirement.

## Open
Open `index.html` in your browser with an internet connection. Chart.js is loaded from a CDN. Press **Add demo reading** several times to populate the three score cards and trend graph. Press **Reset** to clear.

## Connect TrustGuard later
Call the globally available function in `dashboard.js` with real timestamped data:

```javascript
addTrustGuardReading({
  timestamp: '2023-09-22T00:22:00',
  ml_score: 0.15,
  survival_score: 0.08,
  anomaly_score: 0.32
});
```

The numerical scores shown above are examples only. The frontend does **not** assume a score range or thresholds. If the backend score schema differs, update `addTrustGuardReading` to map backend fields.

When an API is available, connect using `fetch` from a hosted frontend and call `addTrustGuardReading(...)` on each received timestamped reading. For cross-origin requests, the backend must permit the frontend origin via CORS.

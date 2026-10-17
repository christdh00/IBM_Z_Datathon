// Frontend-only prototype. Replace addTrustGuardReading(...) calls with API data later.
// No assumptions are made about the TrustGuard score scale or direction.
const MAX_POINTS = 40;
const readings = [];
const elements = {
  ml: document.getElementById('mlScore'),
  survival: document.getElementById('survivalScore'),
  anomaly: document.getElementById('anomalyScore'),
  updated: document.getElementById('lastUpdated'),
};

const chart = new Chart(document.getElementById('scoreChart'), {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'ML', data: [], borderColor: '#61a5fa', backgroundColor: '#61a5fa', tension: 0.25, pointRadius: 3 },
    { label: 'Survival', data: [], borderColor: '#2ed6ad', backgroundColor: '#2ed6ad', tension: 0.25, pointRadius: 3 },
    { label: 'Anomaly', data: [], borderColor: '#ffba68', backgroundColor: '#ffba68', tension: 0.25, pointRadius: 3 },
  ] },
  options: { responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: '#cbd5e1' } } },
    scales: { x: { ticks: { color: '#9aa9c6', maxTicksLimit: 8 }, grid: { color: '#273650' }, title: { display: true, text: 'Timestamp', color: '#9aa9c6' } },
      y: { ticks: { color: '#9aa9c6' }, grid: { color: '#273650' }, title: { display: true, text: 'TrustGuard score', color: '#9aa9c6' } } }
  }
});

function addTrustGuardReading(input) {
  const row = {
    timestamp: new Date(input.timestamp),
    ml: Number(input.ml_score),
    survival: Number(input.survival_score),
    anomaly: Number(input.anomaly_score)
  };
  if (Number.isNaN(row.timestamp.getTime()) || ![row.ml,row.survival,row.anomaly].every(Number.isFinite)) {
    console.warn('Invalid TrustGuard reading', input);
    return;
  }
  readings.push(row);
  if (readings.length > MAX_POINTS) readings.shift();
  elements.ml.textContent = row.ml.toFixed(3);
  elements.survival.textContent = row.survival.toFixed(3);
  elements.anomaly.textContent = row.anomaly.toFixed(3);
  elements.updated.textContent = 'Updated: ' + row.timestamp.toLocaleString();
  chart.data.labels = readings.map(r => r.timestamp.toLocaleTimeString());
  chart.data.datasets[0].data = readings.map(r => r.ml);
  chart.data.datasets[1].data = readings.map(r => r.survival);
  chart.data.datasets[2].data = readings.map(r => r.anomaly);
  chart.update();
}

// Available globally so a future API handler can call this function.
window.addTrustGuardReading = addTrustGuardReading;

let demoIndex = 0;
document.getElementById('demoButton').addEventListener('click', () => {
  const i = demoIndex++;
  addTrustGuardReading({
    timestamp: new Date(Date.now() + i * 1000).toISOString(),
    ml_score: Math.max(0, 0.35 + 0.14 * Math.sin(i / 2) + (Math.random() - 0.5) * 0.08),
    survival_score: Math.max(0, 0.22 + 0.11 * Math.cos(i / 3) + (Math.random() - 0.5) * 0.07),
    anomaly_score: Math.max(0, 0.48 + 0.16 * Math.sin(i / 4) + (Math.random() - 0.5) * 0.07)
  });
});
document.getElementById('resetButton').addEventListener('click', () => {
  readings.length = 0; demoIndex = 0;
  [elements.ml, elements.survival, elements.anomaly].forEach(el => el.textContent = '—');
  elements.updated.textContent = 'No readings yet';
  chart.data.labels = [];
  chart.data.datasets.forEach(dataset => dataset.data = []);
  chart.update();
});

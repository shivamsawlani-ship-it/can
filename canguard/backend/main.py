"""CANGuard backend: replays CAN traffic, scores every frame with the trained model, streams results over a WebSocket."""
import asyncio, random, time
from pathlib import Path
import joblib, numpy as np
import model  # noqa: F401 registers CANForest for joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent
model = joblib.load(ROOT / 'artifacts/model.joblib')
R = joblib.load(ROOT / 'artifacts/replay.joblib')
NAMES = {1: 'DoS', 2: 'Fuzzy', 3: 'Spoof'}
SRC = {'DoS': [R['DoS']], 'Fuzzy': [R['Fuzzy']], 'Spoof': [R['Spoof_gear'], R['Spoof_rpm']]}
for s in [R['Normal'], *[x for v in SRC.values() for x in v]]: s['pos'] = 0
until, use = {t: 0.0 for t in SRC}, {}

app = FastAPI(title='CANGuard')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

def take(s, dt=0.1):
    """Next 100 ms of recorded traffic from a source, using its original timestamps."""
    ts, n = s['ts'], len(s['ts']); p = s['pos'] % n
    q = max(int(np.searchsorted(ts, ts[p] + dt)), p + 1)
    s['pos'] = q % n
    return np.arange(p, q) % n

def tick():
    now = time.time()
    srcs = [use[t] for t in SRC if until[t] > now and t in use] or [R['Normal']]
    parts = [(s, take(s)) for s in srcs]
    X = np.vstack([s['X'][i] for s, i in parts]); y = np.concatenate([s['y'][i] for s, i in parts])
    ids = np.concatenate([s['ids'][i] for s, i in parts]); d = np.vstack([s['d'][i] for s, i in parts])
    t0 = time.perf_counter(); p = model.predict(X); lat = (time.perf_counter() - t0) * 1000 / len(X)
    fl, atk = p > 0, y > 0
    f, nf = np.flatnonzero(fl), np.flatnonzero(~fl)
    sel = list(np.random.permutation(f)[:5]) + list(np.random.permutation(nf)[:3])
    return dict(n=int(len(X)), fl=int(fl.sum()), by={NAMES[k]: int((p == k).sum()) for k in NAMES if (p == k).any()},
                tp=int((fl & atk).sum()), fp=int((fl & ~atk).sum()), fn=int((~fl & atk).sum()), lat_ms=lat,
                sample=[dict(id=int(ids[j]), d=d[j].tolist(), v=NAMES.get(int(p[j])), atk=bool(atk[j])) for j in sel])

@app.post('/inject')
def inject(type: str):
    if type not in SRC: return {'error': 'type must be DoS, Fuzzy or Spoof'}
    s = random.choice(SRC[type])
    onsets = np.flatnonzero(np.diff(np.r_[0, (s['y'] > 0).astype(int)]) == 1)
    s['pos'] = int(random.choice(onsets)) if len(onsets) else 0
    use[type], until[type] = s, time.time() + 4
    return {'ok': True}

@app.websocket('/ws')
async def stream(w: WebSocket):
    await w.accept()
    try:
        while True:
            await w.send_json(tick()); await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass

@app.get('/')
def index():
    return FileResponse(ROOT.parent / 'frontend/index.html')

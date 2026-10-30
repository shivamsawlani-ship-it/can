"""Train the CAN bus intrusion detector and build the replay file.
  python train.py --demo   synthetic data in HCRL format (use this to test the pipeline)
  python train.py          real HCRL CSVs placed in ./data
"""
import argparse, os, time
from collections import deque
import joblib, numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    from model import CANForest

def print_report(y_true, y_pred):
    labels = [0, 1, 2, 3]
    target_names = ['normal', 'DoS', 'Fuzzy', 'Spoof']
    print(f"\n{'classification report:':<20}")
    print(f"{'class':<10} {'precision':<12} {'recall':<12} {'f1-score':<12} {'support':<10}")
    for lbl, name in zip(labels, target_names):
        tp = int(np.sum((y_pred == lbl) & (y_true == lbl)))
        fp = int(np.sum((y_pred == lbl) & (y_true != lbl)))
        fn = int(np.sum((y_pred != lbl) & (y_true == lbl)))
        sup = int(np.sum(y_true == lbl))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{name:<10} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {sup:<10}")

FILES = {'DoS': 'DoS_dataset.csv', 'Fuzzy': 'Fuzzy_dataset.csv',
         'Spoof_gear': 'gear_dataset.csv', 'Spoof_rpm': 'RPM_dataset.csv'}
LABEL = {'DoS': 1, 'Fuzzy': 2, 'Spoof': 3}

def make_demo(folder, secs=60):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    periods = {0x316: .01, 0x153: .01, 0x43F: .01, 0x2C0: .02, 0x130: .02, 0x18F: .01, 0x260: .01, 0x5A0: .02}
    base = {i: rng.integers(0, 256, 8) for i in periods}
    normal = []
    for i, p in periods.items():
        ts = np.arange(0, secs, p)
        ts = ts + rng.normal(0, p * .02, len(ts))
        for t in ts:
            d = base[i].copy(); d[0] = (d[0] + rng.integers(-3, 4)) % 256; d[3] = rng.integers(0, 256)
            normal.append((t, i, d, 'R'))
    kinds = {'DoS_dataset.csv': (0, .0003, lambda: [0] * 8),
             'Fuzzy_dataset.csv': (None, .0005, lambda: rng.integers(0, 256, 8)),
             'gear_dataset.csv': (0x43F, .001, lambda: [1, 69, 96, 255, 107, 0, 0, 0]),
             'RPM_dataset.csv': (0x316, .001, lambda: [5, 125, 255, 128, 0, 0, 0, 0])}
    for fn, (cid, per, pay) in kinds.items():
        ev = list(normal)
        for k in range(0, secs, 10):
            for t in np.arange(k + 3, k + 6, per):
                ev.append((t, cid if cid is not None else int(rng.integers(0, 0x800)), pay(), 'T'))
        ev.sort(key=lambda e: e[0])
        with open(os.path.join(folder, fn), 'w') as f:
            for t, i, d, fl in ev:
                f.write(f"{1478198376 + t:.6f},{i:04x},8,{','.join(f'{int(b):02x}' for b in d)},{fl}\n")

def parse(path, limit):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f):
            if n >= limit: break
            p = line.strip().split(',')
            try:
                dlc = int(p[2]); d = [int(x, 16) for x in p[3:3 + dlc]] + [0] * (8 - dlc)
                rows.append((float(p[0]), int(p[1], 16), dlc, d, p[3 + dlc] == 'T'))
            except (ValueError, IndexError):
                pass
    return rows

def featurize(rows, kind):
    """id, dlc, 8 data bytes, time since last frame, time since last frame with same ID, ID count in last 100 frames."""
    last, lid, win, X, y = None, {}, deque(maxlen=100), [], []
    for ts, cid, dlc, d, atk in rows:
        dt = 0. if last is None else ts - last
        ds = ts - lid.get(cid, ts)
        win.append(cid)
        X.append([cid, dlc, *d, dt, ds, win.count(cid)]); y.append(LABEL[kind] if atk else 0)
        last, lid[cid] = ts, ts
    return np.array(X, float), np.array(y)

def longest_normal(rep):
    best = None
    for k, r in rep.items():
        z = (r['y'] == 0).astype(int)
        idx = np.flatnonzero(np.diff(np.r_[0, z, 0]))
        s, e = idx[::2], idx[1::2]
        if len(s):
            j = int(np.argmax(e - s))
            if best is None or e[j] - s[j] > best[0]: best = (e[j] - s[j], k, s[j], e[j])
    _, k, s, e = best
    return {c: v[s:e] for c, v in rep[k].items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--limit', type=int, default=300000)
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo: make_demo(a.data)
    Xtr, ytr, Xte, yte, rep = [], [], [], [], {}
    for name, fn in FILES.items():
        rows = parse(os.path.join(a.data, fn), a.limit)
        X, y = featurize(rows, name.split('_')[0])
        cut = int(len(X) * .7)  # split by time, not randomly
        Xtr.append(X[:cut]); ytr.append(y[:cut]); Xte.append(X[cut:]); yte.append(y[cut:])
        te = rows[cut:]
        rep[name] = dict(ts=np.array([r[0] for r in te]), X=X[cut:], y=y[cut:],
                         ids=np.array([r[1] for r in te]), d=np.array([r[3] for r in te], dtype=np.uint8))
        print(f'{name}: {len(rows)} frames, {int((y > 0).sum())} attack')
    if SKLEARN_AVAILABLE:
        model = RandomForestClassifier(n_estimators=60, max_depth=14, n_jobs=-1, class_weight='balanced', random_state=0)
        model.fit(np.vstack(Xtr), np.concatenate(ytr))
        Xt, yt = np.vstack(Xte), np.concatenate(yte)
        t0 = time.perf_counter(); p = model.predict(Xt); dt = time.perf_counter() - t0
        print(classification_report(yt, p, labels=[0, 1, 2, 3], target_names=['normal', 'DoS', 'Fuzzy', 'Spoof'], digits=4, zero_division=0))
    else:
        print("Notice: Training pure-NumPy CANForest classifier (OS policy blocked scikit-learn DLLs)")
        model = CANForest(n_estimators=10, max_depth=8, random_state=0)
        model.fit(np.vstack(Xtr), np.concatenate(ytr))
        Xt, yt = np.vstack(Xte), np.concatenate(yte)
        t0 = time.perf_counter(); p = model.predict(Xt); dt = time.perf_counter() - t0
        print_report(yt, p)
    print(f'batch inference: {dt / len(Xt) * 1000:.4f} ms per frame')
    rep['Normal'] = longest_normal(rep)
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/model.joblib'); joblib.dump(rep, 'artifacts/replay.joblib')
    print('saved artifacts/')

if __name__ == '__main__':
    main()

# CANGuard – CAN bus intrusion detection (AI Manthan 2.0, PS8)

## Run it
```
cd backend
pip install -r requirements.txt
python train.py --demo        # synthetic data, checks the pipeline in about a minute
uvicorn main:app --port 8000
```
Open http://localhost:8000 and click the attack buttons.

## Use the real dataset
Download the HCRL Car-Hacking dataset and put DoS_dataset.csv, Fuzzy_dataset.csv, gear_dataset.csv and RPM_dataset.csv in `backend/data/`
(check the file names in your download). Then run `python train.py` (use `--limit 500000` for more rows per file) and restart uvicorn.

## How it works
- `train.py` turns each frame into 13 features (ID, DLC, 8 data bytes, time since last frame, time since last frame with the same ID, ID count in the last 100 frames), splits by time (70/30), trains a Random Forest, and saves the model plus a replay of the test split.
- `main.py` replays the recorded traffic at its original timing, scores every frame, and streams a summary every 100 ms over `/ws`. `POST /inject?type=DoS|Fuzzy|Spoof` switches the replay to a recorded attack burst for 4 seconds.
- `frontend/index.html` draws the dashboard. If the backend is not reachable it falls back to a built-in simulation.

## Be ready to explain to judges
- This is a replay of recorded data, not a live car.
- HCRL attacks are easy to separate, so scores near 100% are expected. Say so before they ask.
- Use one browser tab at a time (all tabs share one replay position).
- The demo data from `--demo` is synthetic and gives perfect scores. Report only results from the real dataset.

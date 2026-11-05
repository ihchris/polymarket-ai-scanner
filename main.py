"""Offline command-line demo and CSV fixture generator."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.collectors.elections import DemoElection
from src.models.bayesian import forecast
from src.storage.database import Database


def main():
    parser = argparse.ArgumentParser(description='Analysis-only Polymarket AI Scanner')
    parser.add_argument('--progress', type=float, default=.5)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--backend', choices=['laplace', 'pymc'], default='laplace')
    parser.add_argument('--export-demo', action='store_true')
    args = parser.parse_args()
    demo = DemoElection()
    if args.export_demo:
        Path('data/demo').mkdir(parents=True, exist_ok=True)
        pd.concat([demo.snapshot(p) for p in [0, .2, .4, .6, .8, 1]]).to_csv('data/demo/partials.csv', index=False)
        Path('data/demo/outcome.json').write_text(json.dumps(demo.outcome(), indent=2))
    snapshot = demo.snapshot(args.progress)
    result = forecast(snapshot, draws=args.draws, backend=args.backend)
    db = Database()
    db.append('election_partials', snapshot.to_dict('records'), timestamp=result.timestamp)
    db.append('forecasts', result.to_dict(), timestamp=result.timestamp, model_version=result.model_version)
    db.close()
    print(json.dumps({'source': 'DEMO — SIMULATED', **result.to_dict()}, indent=2))


if __name__ == '__main__':
    main()

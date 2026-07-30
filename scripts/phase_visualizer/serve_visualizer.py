"""Serve the linecut explorer and run its fitting API locally."""

from __future__ import annotations

import argparse
import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from visualizer_math import fit_linecut

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "phase_visualizer_data.json"
HTML_PATH = HERE / "phase_visualizer.html"
_WORKER_FIELDS = {}


def _field_key(value):
    return str(float(value))


def _config(query):
    return {
        "source": query.get("source", ["smoothed"])[0],
        "loss": query.get("loss", ["soft_l1"])[0],
        "noise": query.get("noise", ["none"])[0],
        "min_points": int(query.get("minPoints", ["9"])[0]),
        "min_span": float(query.get("minSpan", ["1.0"])[0]),
    }


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def _start_worker(data_path):
    global _WORKER_FIELDS
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    dataset = json.loads(Path(data_path).read_text())
    _WORKER_FIELDS = {_field_key(field["field"]): field for field in dataset["fields"]}


def _fit_phase_column(task):
    field_key, index, source, loss, noise, min_points, min_span = task
    result = fit_linecut(
        _WORKER_FIELDS[field_key],
        index,
        source=source,
        loss=loss,
        noise=noise,
        min_points=min_points,
        min_span=min_span,
        include_curves=False,
    )
    return {
        "n": result["localFit"]["n"],
        "nSigma": result["localFit"]["n_sigma"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if not DATA_PATH.exists() or not HTML_PATH.exists():
        parser.error("run build_visualizer.py before starting the server")

    dataset = json.loads(DATA_PATH.read_text())
    fields = {_field_key(field["field"]): field for field in dataset["fields"]}
    pool = (
        ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_start_worker,
            initargs=(str(DATA_PATH),),
        )
        if args.workers > 1
        else None
    )

    @lru_cache(maxsize=1024)
    def cached_fit(
        field_key, linecut_index, source, loss, noise, min_points, min_span, include_curves
    ):
        return fit_linecut(
            fields[field_key],
            linecut_index,
            source=source,
            loss=loss,
            noise=noise,
            min_points=min_points,
            min_span=min_span,
            include_curves=include_curves,
        )

    @lru_cache(maxsize=16)
    def cached_phase(field_key, source, loss, noise, min_points, min_span):
        field = fields[field_key]
        if pool is None:
            columns = []
            for index in range(len(field["linecuts"])):
                result = cached_fit(
                    field_key,
                    index,
                    source,
                    loss,
                    noise,
                    min_points,
                    min_span,
                    False,
                )
                columns.append(
                    {
                        "n": result["localFit"]["n"],
                        "nSigma": result["localFit"]["n_sigma"],
                    }
                )
            return columns

        tasks = [
            (field_key, index, source, loss, noise, min_points, min_span)
            for index in range(len(field["linecuts"]))
        ]
        return list(pool.map(_fit_phase_column, tasks, chunksize=1))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(HERE), **kwargs)

        def _send_json(self, value, status=200):
            payload = _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, error, status=400):
            self._send_json({"error": str(error)}, status)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if not parsed.path.startswith("/api/"):
                if parsed.path == "/":
                    self.path = "/phase_visualizer.html"
                return super().do_GET()

            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/metadata":
                    return self._send_json(
                        {
                            "fields": [
                                {
                                    "field": field["field"],
                                    "fillings": field["fillings"],
                                    "temperatures": field["temperatures"],
                                    "acceptedCount": field["acceptedCount"],
                                }
                                for field in dataset["fields"]
                            ]
                        }
                    )

                field_key = _field_key(query["field"][0])
                field = fields[field_key]
                config = _config(query)

                if parsed.path == "/api/linecut":
                    index = int(query.get("index", ["0"])[0])
                    if not 0 <= index < len(field["linecuts"]):
                        raise ValueError("linecut index is out of range")
                    result = cached_fit(
                        field_key,
                        index,
                        config["source"],
                        config["loss"],
                        config["noise"],
                        config["min_points"],
                        config["min_span"],
                        True,
                    )
                    linecut = field["linecuts"][index]
                    result = {
                        **result,
                        "field": field["field"],
                        "filling": field["fillings"][index],
                        "temperatures": field["temperatures"],
                        "raw": linecut["raw"],
                        "smoothed": linecut["smoothed"],
                        "features": linecut["features"],
                        "range": linecut["range"],
                    }
                    return self._send_json(result)

                if parsed.path == "/api/phase":
                    phase_columns = cached_phase(
                        field_key,
                        config["source"],
                        config["loss"],
                        config["noise"],
                        config["min_points"],
                        config["min_span"],
                    )

                    return self._send_json(
                        {
                            "field": field["field"],
                            "temperatures": field["temperatures"],
                            "fillings": field["fillings"],
                            "heatFillings": field["heatFillings"],
                            "heatmap": field["heatmap"],
                            "logMin": field["logMin"],
                            "logMax": field["logMax"],
                            "phaseColumns": phase_columns,
                            "ranges": [linecut["range"] for linecut in field["linecuts"]],
                            "features": [
                                {**feature, "nu": field["fillings"][index]}
                                for index, linecut in enumerate(field["linecuts"])
                                for feature in linecut["features"]
                            ],
                        }
                    )

                self._error("unknown API endpoint", 404)
            except (KeyError, TypeError, ValueError) as error:
                self._error(error)
            except Exception as error:
                self._error(f"fit failed: {error}", 500)

        def log_message(self, format, *args):
            if not self.path.startswith("/api/"):
                super().log_message(format, *args)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Linecut explorer: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        if pool is not None:
            pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()

import logging
import json
import os
from datetime import datetime
from typing import Any


class FLLogger:
    """
    Centralised logger for the Federated Learning framework.

    Writes structured logs to both the console and a timestamped log file
    under the results directory. Every federated round, client event, and
    aggregation step is logged with a consistent prefix so logs are easy
    to parse later.
    """

    def __init__(self, results_dir: str = "./results", log_level: int = logging.INFO):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(results_dir, f"fl_run_{timestamp}.log")

        # ── Root logger ──────────────────────────────────────────────────────
        self.logger = logging.getLogger("FederatedLearning")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Avoid duplicate handlers if logger is reused
        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(self._get_formatter())
            self.logger.addHandler(ch)

            # File handler
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(log_level)
            fh.setFormatter(self._get_formatter())
            self.logger.addHandler(fh)

        self.log_file_path = log_file

    @staticmethod
    def _get_formatter() -> logging.Formatter:
        return logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # ── Convenience wrappers ─────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    # ── Structured logging helpers ───────────────────────────────────────────

    def section(self, title: str) -> None:
        """Print a prominent section header to separate logical blocks."""
        border = "═" * 60
        self.info(f"\n{border}\n  {title}\n{border}")

    def round_header(self, round_num: int, total_rounds: int) -> None:
        self.info(f"\n{'─' * 60}")
        self.info(f"  ROUND {round_num}/{total_rounds}")
        self.info(f"{'─' * 60}")

    def client_event(self, client_id: int, msg: str) -> None:
        self.info(f"  [Client {client_id}] {msg}")

    def server_event(self, msg: str) -> None:
        self.info(f"  [Server]   {msg}")

    def metrics(self, label: str, value: Any, unit: str = "") -> None:
        self.info(f"  {label:<30}: {value}{unit}")

    def save_json(self, data: dict, path: str) -> None:
        """Persist arbitrary metrics / results as a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        self.info(f"  Metrics saved → {path}")

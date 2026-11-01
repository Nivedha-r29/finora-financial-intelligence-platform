# utils/logger.py
# Simple console logger for the single-client training pipeline.
#
# Keeps all print formatting in one place so the rest of the codebase
# stays clean.  When this project grows into a full Federated Learning
# framework, this logger can be extended with file output and
# structured round-level logging.

def header(title: str) -> None:
    """Print a prominent section header."""
    border = "=" * 40
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}")


def section(title: str) -> None:
    """Print a minor section separator."""
    print(f"\n--- {title} ---")


def info(msg: str) -> None:
    print(msg)


def epoch_log(epoch: int, total: int, loss: float, accuracy: float) -> None:
    print(f"  Epoch {epoch}/{total}  |  Loss: {loss:.4f}  |  Accuracy: {accuracy:.2f}%")


def result(label: str, value) -> None:
    print(f"  {label:<22}: {value}")

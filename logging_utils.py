# modules/logging_utils.py
import logging, sys

def configure_logging(level=logging.INFO):
    # Make console UTF-8 if possible (safe no-op elsewhere)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # override any prior config (Streamlit sometimes configures logging)
    )

    # Also log to a UTF-8 file (great for a packaged app)
    fh = logging.FileHandler("app.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root = logging.getLogger()
    # Avoid duplicate file handlers on reruns
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("app.log")
               for h in root.handlers):
        root.addHandler(fh)

def get_logger(name=None, level=logging.INFO):
    configure_logging(level)
    return logging.getLogger(name or __name__)

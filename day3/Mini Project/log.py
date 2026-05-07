from datetime import datetime

LOG_FILE = "activity.log"


def write_log(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} - {message}\n")


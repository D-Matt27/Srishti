import json
import os

FILE = "users.json"


def load_users():
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r") as f:
        return json.load(f)


def save_users(data):
    with open(FILE, "w") as f:
        json.dump(data, f, indent=4)


def backup_users():
    if os.path.exists(FILE):
        os.rename(FILE, "backup_users.json")
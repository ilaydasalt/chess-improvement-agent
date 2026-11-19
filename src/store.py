"""SQLite persistence: analysed games, profile snapshots, exercises, attempts."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "coach.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    username TEXT, url TEXT, date TEXT, data TEXT,
    PRIMARY KEY (username, url)
);
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT, created_at TEXT, first_game TEXT, last_game TEXT, n_games INTEGER, data TEXT
);
CREATE TABLE IF NOT EXISTS exercises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT, created_at TEXT, skill TEXT, pattern_id TEXT, fen TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exercise_id INTEGER, username TEXT, created_at TEXT, skill TEXT, pattern_id TEXT,
    score REAL, data TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def _u(username: str) -> str:
    return username.strip().lower()


# ---- games ----
def known_game_urls(username: str) -> set[str]:
    with connect() as con:
        return {r[0] for r in con.execute("SELECT url FROM games WHERE username=?", (_u(username),))}


def save_game(username: str, game: dict) -> None:
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO games VALUES (?,?,?,?)",
                    (_u(username), game["url"], game["date"], json.dumps(game)))


def load_games(username: str, urls: list[str] | None = None) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT url, data FROM games WHERE username=? ORDER BY date DESC",
                           (_u(username),)).fetchall()
    games = [json.loads(d) for _, d in rows]
    if urls is not None:
        wanted = set(urls)
        games = [g for g in games if g["url"] in wanted]
    return games


# ---- profiles ----
def save_profile(username: str, profile: dict) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO profiles (username, created_at, first_game, last_game, n_games, data) VALUES (?,?,?,?,?,?)",
            (_u(username), _now(), profile["first_game"], profile["last_game"], profile["n_games"], json.dumps(profile)))
        return cur.lastrowid


def load_profiles(username: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT id, created_at, data FROM profiles WHERE username=? ORDER BY id",
                           (_u(username),)).fetchall()
    out = []
    for pid, created, data in rows:
        p = json.loads(data)
        p["id"], p["created_at"] = pid, created
        out.append(p)
    return out


# ---- exercises & attempts ----
def save_exercise(username: str, ex: dict) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO exercises (username, created_at, skill, pattern_id, fen, data) VALUES (?,?,?,?,?,?)",
            (_u(username), _now(), ex["skill"], ex.get("pattern_id", ""), ex["fen"], json.dumps(ex)))
        return cur.lastrowid


def load_exercises(username: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT id, created_at, data FROM exercises WHERE username=? ORDER BY id DESC",
                           (_u(username),)).fetchall()
    out = []
    for eid, created, data in rows:
        e = json.loads(data)
        e["id"], e["created_at"] = eid, created
        out.append(e)
    return out


def save_attempt(username: str, exercise: dict, score: float, data: dict) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO attempts (exercise_id, username, created_at, skill, pattern_id, score, data) VALUES (?,?,?,?,?,?,?)",
            (exercise["id"], _u(username), _now(), exercise["skill"], exercise.get("pattern_id", ""), score, json.dumps(data)))


def load_attempts(username: str) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT exercise_id, created_at, skill, pattern_id, score, data FROM attempts WHERE username=? ORDER BY id",
            (_u(username),)).fetchall()
    return [{"exercise_id": e, "created_at": c, "skill": s, "pattern_id": p, "score": sc, **json.loads(d)}
            for e, c, s, p, sc, d in rows]

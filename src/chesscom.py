from urllib.parse import urlparse
import requests

BASE = "https://api.chess.com/pub"
HEADERS = {"User-Agent": "ChessImprovementAgent/0.1 (personal chess analysis project)"}
import time

def fetch_recent_pgn(username: str, months: int = 3) -> str:
    archives = fetch_archives(username)[-months:]   # son N ay
    pgns = []
    for url in archives:
        try:
            pgn = fetch_month_pgn(url).strip()
        except requests.HTTPError as e:
            print(f"Atlandı {url}: {e}")
            continue
        if pgn:
            pgns.append(pgn)
        time.sleep(0.3)  # API'ye nazik ol
    if not pgns:
        raise ValueError("Seçilen aylarda oyun bulunamadı.")
    return "\n\n".join(pgns)

def fetch_recent_games(username: str, time_class: str = "rapid", count: int = 100,
                       progress_cb=None) -> list[str]:
    """Return PGNs of the newest `count` standard-chess games of `time_class`, newest first.

    Walks the monthly archives backwards until enough games are collected.
    """
    archives = fetch_archives(username)
    games: list[dict] = []
    for i, url in enumerate(reversed(archives), start=1):
        try:
            month_games = _get(url).json().get("games", [])
        except requests.HTTPError as e:
            print(f"Skipped {url}: {e}")
            continue
        games += [g for g in month_games
                  if g.get("time_class") == time_class and g.get("rules") == "chess" and g.get("pgn")]
        if progress_cb:
            progress_cb(min(len(games), count), count, i)
        if len(games) >= count:
            break
        time.sleep(0.3)  # be polite to the API
    games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
    return [g["pgn"].strip() for g in games[:count]]


def _get(url: str):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response

def fetch_archives(username: str) -> list[str]:
    clean = username.strip().lower()
    if not clean or any(ch in clean for ch in "/?# "):
        raise ValueError("Please enter a valid Chess.com username.")
    data = _get(f"{BASE}/player/{clean}/games/archives").json()
    archives = data.get("archives", [])
    if not archives:
        raise ValueError("No public game archives were found for this username.")
    return archives


def fetch_month_pgn(archive_url: str) -> str:
    parsed = urlparse(archive_url)
    parts = parsed.path.strip("/").split("/")

    if len(parts) < 6 or parts[-3] != "games":
        raise ValueError(f"Unexpected Chess.com archive URL: {archive_url}")

    username = parts[-4]
    year = parts[-2]
    month = parts[-1]

    url = f"{BASE}/player/{username}/games/{year}/{month}/pgn"

    return _get(url).text

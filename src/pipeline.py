"""Fetch → analyse (incrementally) → profile snapshot."""
from io import StringIO

import chess.pgn

from src import store
from src.analyzer import analyze_game, open_engine, game_key
from src.chesscom import fetch_recent_games
from src.profile import build_profile


def update_player(username: str, engine_path: str, depth: int = 12, count: int = 100,
                  time_class: str = "rapid", progress=None) -> dict:
    """Analyse the newest `count` games (only those not analysed before) and
    store a new profile snapshot built from exactly those games."""
    say = progress or (lambda frac, text: None)
    say(0.0, "Fetching games from Chess.com…")
    pgns = fetch_recent_games(username, time_class=time_class, count=count)
    if not pgns:
        raise ValueError(f"No {time_class} games found for {username}.")

    known = store.known_game_urls(username)
    parsed = []
    for pgn in pgns:
        g = chess.pgn.read_game(StringIO(pgn))
        if g is not None and not g.errors:
            parsed.append(g)
    urls = [game_key(g) for g in parsed]
    todo = [g for g in parsed if game_key(g) not in known]

    if todo:
        engine = open_engine(engine_path)
        try:
            for i, g in enumerate(todo, start=1):
                say(i / len(todo), f"Analysing game {i}/{len(todo)} (new games only)…")
                store.save_game(username, analyze_game(engine, g, username, depth))
        finally:
            engine.quit()

    games = store.load_games(username, urls)
    profile = build_profile(username, games)
    profile["game_urls"] = urls
    profile["depth"] = depth
    profile["id"] = store.save_profile(username, profile)
    say(1.0, f"Profile built from {len(games)} games ({len(todo)} newly analysed).")
    return profile

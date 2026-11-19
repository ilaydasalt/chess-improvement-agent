"""Personalised exercises built from the player's own critical positions.

Most exercises come from positions where a detected pattern produced an error
("can you find it now, and explain why?"). A few *control* positions come from
the same strategic context where the player did fine; comparing both tells us
whether a pattern is a real thinking gap or just noise."""
import random

import chess
import chess.engine

from src import store
from src.analyzer import engine_lines, open_engine, MATE
from src.coach import make_question
from src.profile import STRUCTURE_TAGS, COORD_TAGS, _all_moves, _select, weakest_skills, pattern_matcher

ERROR_TO_SKILL = {
    "missed_threat": "threat_awareness",
    "strategic": "strategic_planning",
    "missed_opportunity": "positional_evaluation",
    "conversion": "advantage_conversion",
}


def skill_for_pattern(p: dict) -> str:
    pid = p["id"]
    tag = pid.split(":", 1)[1]
    if tag in STRUCTURE_TAGS or tag == "pawn_moves":
        return "pawn_structure"
    if tag in COORD_TAGS:
        return "piece_coordination"
    if tag in ("winning", "blown_wins"):
        return "advantage_conversion"
    if tag == "losing":
        return "defense"
    if pid in ("phase:endgame", "phase:opening"):
        return tag
    return ERROR_TO_SKILL.get(p["dominant_error"], "strategic_planning")


def _exercise_from(m: dict, skill: str, pattern: dict | None, control: bool) -> dict:
    return {
        "fen": m["fen_before"] if "fen_before" in m else m["fen"],
        "skill": skill,
        "pattern_id": pattern["id"] if pattern else "",
        "pattern_label": pattern["label"] if pattern else "",
        "tags": m["tags"], "played": m["played"], "best": m["best"], "loss": m["loss"],
        "refutation": m.get("refutation", ""), "game_url": m.get("game_url", ""),
        "game_date": m.get("game_date", ""), "control": control,
    }


def generate_exercises(username: str, profile: dict, games: list[dict], engine_path: str,
                       api_key: str | None, n: int = 5, depth: int = 14) -> list[dict]:
    used = {e["fen"] for e in store.load_exercises(username)}
    moves = _all_moves(games)
    candidates: list[dict] = []

    # 1) Error positions from detected patterns (strongest hypotheses first).
    for p in profile["patterns"]:
        skill = skill_for_pattern(p)
        for ex in p["examples"]:
            if ex["fen"] not in used:
                candidates.append(_exercise_from({**ex, "fen_before": ex["fen"]}, skill, p, control=False))
                used.add(ex["fen"])
                break
    # 2) Worst positions for the weakest skills.
    for skill in weakest_skills(profile):
        pool = sorted((m for m in _select(moves, skill) if m["loss"] >= 100 and m["fen_before"] not in used),
                      key=lambda m: -m["loss"])
        if pool:
            candidates.append(_exercise_from(pool[0], skill, None, control=False))
            used.add(pool[0]["fen_before"])

    # 3) One control position: same context as the top pattern, but the player played well.
    control = None
    if profile["patterns"]:
        p = profile["patterns"][0]
        match = pattern_matcher(p["id"])
        ok = [m for m in moves if match(m) and m["loss"] < 20 and not m["best_forcing"]
              and m["fen_before"] not in used and m["phase"] != "opening"]
        if ok:
            control = _exercise_from(random.choice(ok), skill_for_pattern(p), p, control=True)

    n_errors = n - 1 if control else n
    engine = open_engine(engine_path)
    chosen = []
    try:
        def prepare(ex) -> bool:
            ex["lines"] = engine_lines(engine, ex["fen"], depth=depth, multipv=3)
            if not ex["lines"]:
                return False
            ex["best"] = ex["lines"][0]["move"]
            if ex["control"]:
                return True
            # Deeper search says the game move was close to best: not a real error.
            top = ex["lines"][0]["eval"]
            return ex["played"] not in {l["move"] for l in ex["lines"] if top - l["eval"] < 50}

        for ex in candidates:
            if len(chosen) >= n_errors:
                break
            if prepare(ex):
                chosen.append(ex)
        if control and prepare(control):
            chosen.append(control)
    finally:
        engine.quit()

    random.shuffle(chosen)  # don't reveal which one is the control position
    for ex in chosen:
        ex["question"] = make_question(ex, api_key)
        ex["id"] = store.save_exercise(username, ex)
    return chosen


def evaluate_move(engine_path: str, fen: str, san: str, depth: int = 14) -> tuple[str | None, int | None]:
    """Return (normalised SAN, eval for the mover) or (None, None) if illegal."""
    board = chess.Board(fen)
    try:
        move = board.parse_san(san.strip())
    except ValueError:
        return None, None
    norm = board.san(move)
    mover = board.turn
    board.push(move)
    engine = open_engine(engine_path)
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
    finally:
        engine.quit()
    return norm, info["score"].pov(mover).score(mate_score=MATE)

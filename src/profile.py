"""Player profile + strategic pattern detection from analysed games.

Everything here is a *hypothesis* derived from engine numbers. PGNs don't show
what the player was thinking, so each skill score and pattern carries a
confidence level and is meant to be confirmed or rejected by exercises."""
from collections import defaultdict
from statistics import mean

from src.analyzer import classify, wp_loss
from src.features import TAG_LABELS, eval_tag

ERR = 80  # centipawn loss counted as a real error for pattern statistics

SKILLS = {
    "positional_evaluation": "Positional evaluation",
    "strategic_planning": "Strategic planning",
    "pawn_structure": "Pawn structure",
    "piece_coordination": "Piece coordination",
    "threat_awareness": "Anticipating opponent threats",
    "advantage_conversion": "Converting an advantage",
    "defense": "Defending worse positions",
    "endgame": "Endgame technique",
    "opening": "Opening play",
}

SKILL_HELP = {
    "positional_evaluation": "Quiet positions (no forcing best move): how well you pick the right non-tactical move.",
    "strategic_planning": "Balanced middlegames with a quiet best move: choosing a sound plan.",
    "pawn_structure": "Pawn moves and positions with notable pawn structures (IQP, doubled, passed, locked centre).",
    "piece_coordination": "Positions with loose pieces, undeveloped pieces or low mobility.",
    "threat_awareness": "How often your move allowed an immediate forcing blow (capture/check) by the opponent.",
    "advantage_conversion": "Positions where you were ≥ +1.5, and how often clearly winning games were won.",
    "defense": "Positions where you were ≤ −1.5.",
    "endgame": "Endgame positions.",
    "opening": "Opening phase (first ~10 moves).",
}

STRUCTURE_TAGS = {"isolated_pawn", "iqp", "doubled_pawns", "own_passed_pawn", "opp_passed_pawn", "closed_center"}
COORD_TAGS = {"loose_piece", "low_mobility", "undeveloped"}


def _acpl_score(moves) -> float | None:
    if not moves:
        return None
    avg = mean(min(m["loss"], 300) for m in moves)
    return round(max(0.0, min(100.0, 100 - avg * 0.8)), 1)


def _confidence(n: int) -> str:
    return "low" if n < 30 else "medium" if n < 100 else "high"


def _select(moves, skill):
    if skill == "positional_evaluation":
        return [m for m in moves if m["phase"] != "opening" and not m["best_forcing"]]
    if skill == "strategic_planning":
        return [m for m in moves if m["phase"] == "middlegame" and abs(m["eval_before"]) < 150 and not m["best_forcing"]]
    if skill == "pawn_structure":
        return [m for m in moves if m["phase"] != "opening" and (m["pawn_move"] or STRUCTURE_TAGS & set(m["tags"]))]
    if skill == "piece_coordination":
        return [m for m in moves if COORD_TAGS & set(m["tags"])]
    if skill == "advantage_conversion":
        return [m for m in moves if m["eval_before"] >= 150 and m["eval_before"] < 2000]
    if skill == "defense":
        return [m for m in moves if m["eval_before"] <= -150 and m["eval_before"] > -2000]
    if skill in ("endgame", "opening"):
        return [m for m in moves if m["phase"] == skill]
    return moves


def _won(game) -> bool:
    return (game["your_color"] == "White" and game["result"] == "1-0") or \
           (game["your_color"] == "Black" and game["result"] == "0-1")


def _all_moves(games):
    out = []
    for g in games:
        for m in g["moves"]:
            # Recompute loss as expected-score drop (older records stored raw cp loss).
            loss = 0 if m["loss"] == 0 else wp_loss(m["eval_before"], m["eval_after"])
            out.append({**m, "loss": loss, "category": classify(loss),
                        "error_type": m["error_type"] if loss >= 80 else None,
                        "game_url": g["url"], "game_date": g["date"],
                        "tags": m["tags"] + [eval_tag(m["eval_before"])]})
    return out


def build_profile(username: str, games: list[dict]) -> dict:
    moves = _all_moves(games)
    skills = {}
    for key, label in SKILLS.items():
        if key == "threat_awareness":
            n = len(moves)
            missed = sum(1 for m in moves if m["error_type"] == "missed_threat")
            per100 = 100 * missed / n if n else 0
            score = round(max(0.0, 100 - per100 * 8), 1) if n else None
            skills[key] = {"label": label, "score": score, "n": n, "confidence": _confidence(n),
                           "detail": f"{missed} moves allowed a forcing reply ({per100:.1f} per 100 moves)"}
            continue
        sel = _select(moves, key)
        score = _acpl_score(sel)
        detail = ""
        if sel:
            errs = sum(1 for m in sel if m["loss"] >= ERR)
            detail = f"avg loss {mean(min(m['loss'], 300) for m in sel):.0f}, error rate {100 * errs / len(sel):.0f}%"
        if key == "advantage_conversion":
            winning_games = [g for g in games if any(m["eval_before"] >= 300 for m in g["moves"])]
            if winning_games:
                conv = sum(_won(g) for g in winning_games) / len(winning_games)
                detail += f"; won {conv:.0%} of {len(winning_games)} games where you reached ≥ +3"
                if score is not None:
                    score = round(0.5 * score + 50 * conv, 1)
        skills[key] = {"label": label, "score": score, "n": len(sel), "confidence": _confidence(len(sel)),
                       "detail": detail}

    dates = sorted(g["date"] for g in games)
    elos = [g["your_elo"] for g in sorted(games, key=lambda g: g["date"]) if g.get("your_elo")]
    wins = sum(_won(g) for g in games)
    draws = sum(g["result"] == "1/2-1/2" for g in games)
    return {
        "username": username,
        "n_games": len(games),
        "n_moves": len(moves),
        "first_game": dates[0] if dates else "",
        "last_game": dates[-1] if dates else "",
        "elo": elos[-1] if elos else None,
        "record": {"wins": wins, "draws": draws, "losses": len(games) - wins - draws},
        "skills": skills,
        "error_types": _error_type_counts(moves),
        "patterns": detect_patterns(moves, games),
    }


def _error_type_counts(moves):
    c = defaultdict(int)
    for m in moves:
        if m["error_type"]:
            c[m["error_type"]] += 1
    return dict(c)


def _example(m):
    return {"fen": m["fen_before"], "played": m["played"], "best": m["best"], "loss": m["loss"],
            "eval_before": m["eval_before"], "game_url": m["game_url"], "game_date": m["game_date"],
            "ply": m["ply"], "refutation": m["refutation"], "error_type": m["error_type"], "tags": m["tags"]}


def pattern_matcher(pid: str):
    """Predicate selecting the positions a pattern id refers to."""
    kind, key = pid.split(":", 1)
    if kind == "tag":
        return lambda m: key in m["tags"]
    if kind == "phase":
        return lambda m: m["phase"] == key
    if key == "pawn_moves":
        return lambda m: m["pawn_move"] and not m["best_forcing"] and m["phase"] != "opening"
    if key == "quiet_best":
        return lambda m: not m["best_forcing"] and m["phase"] != "opening"
    return lambda m: m["eval_before"] >= 300  # blown_wins


def detect_patterns(moves: list[dict], games: list[dict], min_positions: int = 20, min_games: int = 4) -> list[dict]:
    """Strategic contexts in which the player errs clearly more often than usual,
    across several different games. Each pattern is a hypothesis to test."""
    if not moves:
        return []
    base_rate = sum(m["loss"] >= ERR for m in moves) / len(moves)
    patterns = []

    def consider(pid, label, subset, kind):
        game_set = {m["game_url"] for m in subset}
        if len(subset) < min_positions or len(game_set) < min_games:
            return
        errs = [m for m in subset if m["loss"] >= ERR]
        rate = len(errs) / len(subset)
        err_games = {m["game_url"] for m in errs}
        if rate < base_rate * 1.3 or rate - base_rate < 0.05 or len(err_games) < 3:
            return
        types = defaultdict(int)
        for m in errs:
            types[m["error_type"] or "strategic"] += 1
        dominant = max(types, key=types.get)
        patterns.append({
            "id": pid, "kind": kind, "label": label,
            "n_positions": len(subset), "n_games": len(game_set), "n_error_games": len(err_games),
            "error_rate": round(rate, 3), "baseline": round(base_rate, 3),
            "ratio": round(rate / base_rate, 2) if base_rate else None,
            "dominant_error": dominant, "error_types": dict(types),
            "examples": [_example(m) for m in sorted(errs, key=lambda m: -m["loss"])[:6]],
        })

    def sel(pid):
        f = pattern_matcher(pid)
        return [m for m in moves if f(m)]

    for t in sorted({t for m in moves for t in m["tags"]}):
        consider(f"tag:{t}", TAG_LABELS.get(t, t), sel(f"tag:{t}"), "context")
    for ph in ("opening", "middlegame", "endgame"):
        consider(f"phase:{ph}", f"{ph} phase", sel(f"phase:{ph}"), "phase")
    consider("behavior:pawn_moves", "pawn moves in quiet positions", sel("behavior:pawn_moves"), "behavior")
    consider("behavior:quiet_best", "positions where the best move was quiet (no capture/check)",
             sel("behavior:quiet_best"), "behavior")

    # Advantage not converted: a game-level behaviour.
    winning = [g for g in games if any(m["eval_before"] >= 300 for m in g["moves"])]
    blown = [g for g in winning if not _won(g)]
    if len(winning) >= 4 and len(blown) / len(winning) >= 0.3:
        ex = []
        for g in blown:
            seen_win = False
            for m in g["moves"]:
                seen_win = seen_win or m["eval_before"] >= 300
                if seen_win and m["loss"] and wp_loss(m["eval_before"], m["eval_after"]) >= 150:
                    ex.append(_example({**m, "game_url": g["url"], "game_date": g["date"]}))
                    break
        patterns.append({
            "id": "behavior:blown_wins", "kind": "behavior",
            "label": "winning positions (≥ +3) that were not won",
            "n_positions": len(winning), "n_games": len(winning), "n_error_games": len(blown),
            "error_rate": round(len(blown) / len(winning), 3), "baseline": None, "ratio": None,
            "dominant_error": "conversion", "error_types": {}, "examples": ex[:6],
        })

    patterns.sort(key=lambda p: -(p["ratio"] or 2) * p["n_error_games"])
    return patterns[:10]


def verification(attempts: list[dict], key: str, field: str) -> dict:
    """Exercise evidence for a skill or a pattern."""
    xs = [a["score"] for a in attempts if a.get(field) == key]
    if not xs:
        return {"status": "untested", "n": 0, "avg": None}
    avg = mean(xs)
    if len(xs) < 3:
        status = "testing"
    elif avg < 0.5:
        status = "confirmed weakness"
    elif avg >= 0.75:
        status = "not confirmed"
    else:
        status = "mixed"
    return {"status": status, "n": len(xs), "avg": round(avg, 2)}


def weakest_skills(profile: dict, k: int = 3) -> list[str]:
    scored = [(v["score"], key) for key, v in profile["skills"].items()
              if v["score"] is not None and v["n"] >= 15]
    return [key for _, key in sorted(scored)[:k]]

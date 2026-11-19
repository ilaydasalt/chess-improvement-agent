"""Stockfish analysis that records, for every move the player makes, the
engine numbers plus strategic context tags. These per-move records are the
raw material for the profile, pattern detection and training exercises."""
import asyncio
import math
import sys
from io import StringIO

import chess
import chess.engine
import chess.pgn

from src.features import position_tags, phase_of, is_forcing

MATE = 10000


def open_engine(engine_path: str):
    # Streamlit on Windows uses the Selector event loop, which cannot spawn
    # subprocesses (NotImplementedError). Use the Proactor loop while starting the engine.
    if sys.platform == "win32":
        old_policy = asyncio.get_event_loop_policy()
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            return chess.engine.SimpleEngine.popen_uci(engine_path)
        finally:
            asyncio.set_event_loop_policy(old_policy)
    return chess.engine.SimpleEngine.popen_uci(engine_path)


def win_prob(cp: int) -> float:
    """Expected score (0..1) for a centipawn eval (Lichess model)."""
    return 1 / (1 + math.exp(-0.00368208 * cp))


def wp_loss(eval_before: int, eval_after: int) -> int:
    """Drop in expected score ×1000. Near equality ≈ centipawns, but a mate
    turning into +6 no longer counts as a huge 'loss'."""
    return max(0, round(1000 * (win_prob(eval_before) - win_prob(eval_after))))


def _cp(info, pov) -> int:
    return info["score"].pov(pov).score(mate_score=MATE)


def classify(loss: int) -> str | None:
    if loss >= 300:
        return "blunder"
    if loss >= 100:
        return "mistake"
    if loss >= 50:
        return "inaccuracy"
    return None


def _error_type(board_before: chess.Board, best: chess.Move | None, refutation: chess.Move | None,
                board_after: chess.Board, eval_before: int, loss: int) -> str | None:
    """Rough cause of a costly move. Heuristic, not ground truth."""
    if loss < 80:
        return None
    if refutation is not None and is_forcing(board_after, refutation):
        return "missed_threat"          # opponent now has a forcing blow the player didn't see
    if best is not None and is_forcing(board_before, best):
        return "missed_opportunity"     # player had a forcing move and missed it
    return "strategic"                  # quiet position, quiet best move: plan / evaluation error


def _user_color(game, username: str) -> chess.Color:
    u = (username or "").lower()
    if game.headers.get("Black", "").lower() == u:
        return chess.BLACK
    return chess.WHITE


def analyze_game(engine, game, username: str, depth: int) -> dict:
    """Analyse one game. Each position is searched once; the eval after a move
    is reused as the eval before the next one."""
    color = _user_color(game, username)
    board = game.board()
    limit = chess.engine.Limit(depth=depth)
    info = engine.analyse(board, limit)
    moves_out = []
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        mover = board.turn
        before = board.copy(stack=False)
        eval_before = _cp(info, mover)
        best = info.get("pv", [None])[0]
        san = board.san(move)
        board.push(move)
        info = engine.analyse(board, limit)
        if mover != color:
            continue
        eval_after = _cp(info, mover)
        # Playing the engine's own choice is never an error (avoids search-noise "losses").
        loss = 0 if move == best else wp_loss(eval_before, eval_after)
        refutation = info.get("pv", [None])[0]
        rec = {
            "ply": ply,
            "fen_before": before.fen(),
            "played": san,
            "best": before.san(best) if best else "",
            "refutation": board.san(refutation) if refutation else "",
            "eval_before": eval_before,
            "eval_after": eval_after,
            "loss": loss,
            "category": classify(loss),
            "error_type": _error_type(before, best, refutation, board, eval_before, loss),
            "best_forcing": bool(best is not None and is_forcing(before, best)),
            "phase": phase_of(before),
            "pawn_move": before.piece_type_at(move.from_square) == chess.PAWN,
            "tags": position_tags(before, color),
        }
        moves_out.append(rec)

    h = game.headers
    return {
        "url": h.get("Link", "") or f"{h.get('White')}-{h.get('Black')}-{h.get('UTCDate', h.get('Date'))}-{h.get('UTCTime', '')}",
        "date": (h.get("UTCDate") or h.get("Date", "")).replace(".", "-"),
        "white": h.get("White", "?"),
        "black": h.get("Black", "?"),
        "result": h.get("Result", "*"),
        "time_control": h.get("TimeControl", ""),
        "eco": h.get("ECO", "") or h.get("ECOUrl", "").rsplit("/", 1)[-1],
        "your_color": "White" if color == chess.WHITE else "Black",
        "your_elo": int(h.get("WhiteElo" if color == chess.WHITE else "BlackElo", 0) or 0),
        "moves": moves_out,
    }


def iter_games(pgn_text: str):
    stream = StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        if not game.errors:
            yield game


def game_key(game) -> str:
    h = game.headers
    return h.get("Link", "") or f"{h.get('White')}-{h.get('Black')}-{h.get('UTCDate', h.get('Date'))}-{h.get('UTCTime', '')}"


def engine_lines(engine, fen: str, depth: int = 14, multipv: int = 3) -> list[dict]:
    """Top engine candidate moves with short principal variations (from side to move)."""
    board = chess.Board(fen)
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    out = []
    for inf in infos:
        pv = inf.get("pv", [])[:6]
        b = board.copy()
        sans = []
        for m in pv:
            sans.append(b.san(m))
            b.push(m)
        out.append({"move": sans[0] if sans else "", "line": " ".join(sans), "eval": _cp(inf, board.turn)})
    return out

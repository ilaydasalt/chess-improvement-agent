"""Strategic context tags for a position, from the player's point of view.

These are cheap, explainable heuristics. They describe *what kind of
position* a decision was made in, so that errors can be grouped by
strategic theme instead of being listed one by one."""
import chess

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

# Human-readable labels for tags (used in the UI and in prompts).
TAG_LABELS = {
    "isolated_pawn": "you have an isolated pawn",
    "iqp": "isolated queen's pawn (IQP) structure",
    "doubled_pawns": "you have doubled pawns",
    "own_passed_pawn": "you have a passed pawn",
    "opp_passed_pawn": "opponent has a passed pawn",
    "open_center": "open centre (no d/e pawns)",
    "closed_center": "closed / locked centre",
    "opposite_castling": "opposite-side castling",
    "king_exposed": "your king's pawn shield is weakened",
    "king_uncastled": "your king is still uncastled (after move 12)",
    "undeveloped": "minor pieces still undeveloped (after move 12)",
    "loose_piece": "you have an undefended piece under attack",
    "low_mobility": "your pieces have clearly less mobility than the opponent's",
    "bishop_pair": "you have the bishop pair",
    "opp_bishop_pair": "opponent has the bishop pair",
    "queens_off": "queens are off the board",
    "winning": "you are clearly better (≥ +1.5)",
    "losing": "you are clearly worse (≤ −1.5)",
    "balanced": "roughly balanced position",
}

PHASES = ("opening", "middlegame", "endgame")


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values()
               if p.color == color and p.piece_type in PIECE_VALUE)


def phase_of(board: chess.Board) -> str:
    non_pawn = sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values()
                   if p.piece_type not in (chess.PAWN, chess.KING))
    if board.fullmove_number <= 10 and non_pawn >= 50:
        return "opening"
    if non_pawn <= 26:
        return "endgame"
    return "middlegame"


def is_forcing(board: chess.Board, move: chess.Move) -> bool:
    """Capture, check or promotion."""
    return board.is_capture(move) or board.gives_check(move) or move.promotion is not None


def _pawn_files(board, color):
    files = {}
    for sq in board.pieces(chess.PAWN, color):
        files.setdefault(chess.square_file(sq), []).append(sq)
    return files


def _isolated(board, color) -> list[int]:
    files = _pawn_files(board, color)
    return [f for f in files if (f - 1) not in files and (f + 1) not in files]


def _passed(board, color) -> int:
    opp = board.pieces(chess.PAWN, not color)
    n = 0
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for o in opp:
            of, orank = chess.square_file(o), chess.square_rank(o)
            if abs(of - f) <= 1 and ((orank > r) if color == chess.WHITE else (orank < r)):
                blocked = True
                break
        n += not blocked
    return n


def _king_shield(board, color) -> int:
    k = board.king(color)
    if k is None:
        return 3
    f, r = chess.square_file(k), chess.square_rank(k)
    step = 1 if color == chess.WHITE else -1
    cnt = 0
    for df in (-1, 0, 1):
        for dr in (1, 2):
            ff, rr = f + df, r + step * dr
            if 0 <= ff < 8 and 0 <= rr < 8:
                p = board.piece_at(chess.square(ff, rr))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    cnt += 1
                    break
    return cnt


def _castled_side(board, color):
    k = board.king(color)
    if k is None:
        return None
    f = chess.square_file(k)
    home = 0 if color == chess.WHITE else 7
    if chess.square_rank(k) != home:
        return None
    return "k" if f >= 6 else "q" if f <= 2 else None


def _mobility(board, color) -> int:
    b = board.copy(stack=False)
    b.turn = color
    return sum(1 for m in b.legal_moves if b.piece_type_at(m.from_square) != chess.PAWN)


def position_tags(board: chess.Board, color: chess.Color, eval_cp: int | None = None) -> list[str]:
    tags = []
    opp = not color
    iso = _isolated(board, color)
    if iso:
        tags.append("isolated_pawn")
        if 3 in iso and len(board.pieces(chess.PAWN, color)) >= 4:
            tags.append("iqp")
    if any(len(v) > 1 for v in _pawn_files(board, color).values()):
        tags.append("doubled_pawns")
    if _passed(board, color):
        tags.append("own_passed_pawn")
    if _passed(board, opp):
        tags.append("opp_passed_pawn")

    center_pawns = [sq for sq in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
                    if chess.square_file(sq) in (3, 4)]
    if not center_pawns:
        tags.append("open_center")
    else:
        locked = 0
        for sq in board.pieces(chess.PAWN, chess.WHITE):
            if chess.square_file(sq) in (3, 4) and sq + 8 < 64:
                p = board.piece_at(sq + 8)
                if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                    locked += 1
        if locked >= 2:
            tags.append("closed_center")

    mine, theirs = _castled_side(board, color), _castled_side(board, opp)
    if mine and theirs and mine != theirs:
        tags.append("opposite_castling")
    if board.fullmove_number > 12 and mine is None and board.queens:
        tags.append("king_uncastled")
    if mine and _king_shield(board, color) <= 1 and board.queens:
        tags.append("king_exposed")

    if board.fullmove_number > 12:
        home = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
        minors = (board.pieces_mask(chess.KNIGHT, color) | board.pieces_mask(chess.BISHOP, color)) & home
        if chess.popcount(minors) >= 2:
            tags.append("undeveloped")

    for sq, p in board.piece_map().items():
        if p.color == color and p.piece_type not in (chess.PAWN, chess.KING):
            if board.is_attacked_by(opp, sq) and not board.is_attacked_by(color, sq):
                tags.append("loose_piece")
                break

    m_me, m_opp = _mobility(board, color), _mobility(board, opp)
    if m_opp >= 10 and m_me < 0.7 * m_opp:
        tags.append("low_mobility")

    if len(board.pieces(chess.BISHOP, color)) >= 2:
        tags.append("bishop_pair")
    if len(board.pieces(chess.BISHOP, opp)) >= 2:
        tags.append("opp_bishop_pair")
    if not board.queens:
        tags.append("queens_off")
    return tags


def eval_tag(eval_cp: int) -> str:
    if eval_cp >= 150:
        return "winning"
    if eval_cp <= -150:
        return "losing"
    return "balanced"

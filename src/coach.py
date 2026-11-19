"""AI Chess Coach: profile narrative, Socratic position coaching, exercise
questions and answer grading. Uses Claude when an API key is available and
falls back to rule-based text otherwise, so the app always works."""
import json
import os

from src.features import TAG_LABELS

MODEL = "claude-opus-5-5"

COACH_SYSTEM = """You are an analytical chess coach working with one specific player.
You know their profile (engine-derived skill estimates and recurring strategic patterns), but treat
every pattern as a hypothesis: game records do not show what the player was thinking.

How you coach:
- Interpret the position in words first: pawn structure, piece activity, king safety, who has the plan.
- Explain plans and alternatives, not just the best move. Never open with the engine's top move.
- Ask the player one or two thought-provoking questions per reply and react to their reasoning.
- Pitch the language to the player's rating. Be concise (under ~180 words) unless asked for depth.
- Use the engine data you are given as ground truth for concrete lines; do not invent variations.
- When the player's answer reveals how they think, say what it suggests about the hypothesis.
- Reply in the language the player writes in."""


def _client(api_key: str | None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def available(api_key: str | None) -> bool:
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _call(api_key, system, messages, max_tokens=4000, schema=None, effort="medium") -> str:
    import anthropic
    client = _client(api_key)
    output_config = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    try:
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system, messages=messages,
            output_config=output_config,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Claude API key is invalid.")
    except anthropic.RateLimitError:
        raise RuntimeError("Claude API rate limit reached; try again in a minute.")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise RuntimeError("Could not reach the Claude API (network).")
    if resp.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    return "".join(b.text for b in resp.content if b.type == "text")


def _profile_brief(profile: dict) -> dict:
    """Compact profile for prompts (no example lists)."""
    return {
        "elo": profile.get("elo"), "games": profile["n_games"], "record": profile["record"],
        "skills": {k: {"score": v["score"], "n": v["n"], "confidence": v["confidence"], "detail": v["detail"]}
                   for k, v in profile["skills"].items()},
        "error_types": profile["error_types"],
        "patterns": [{k: p[k] for k in ("id", "label", "error_rate", "baseline", "n_games", "n_error_games",
                                        "dominant_error")} for p in profile["patterns"]],
    }


# ---------------- Player profile narrative ----------------
def profile_narrative(profile: dict, api_key: str | None, verification: dict | None = None) -> str:
    if not available(api_key):
        return _fallback_narrative(profile)
    prompt = (
        "Write this player's strategic profile in Turkish, as markdown with these sections: "
        "**Güçlü yönler**, **Gelişime açık alanlar**, **Tekrarlayan stratejik örüntüler (hipotez)**, "
        "**Doğrulanması gerekenler** (which hypotheses the exercises should test and how), "
        "**Önerilen 2 haftalık odak**. Be specific, cite the numbers, flag low-confidence estimates, "
        "and keep it under 400 words.\n\n"
        f"Profile data:\n{json.dumps(_profile_brief(profile), ensure_ascii=False)}\n\n"
        f"Exercise evidence so far:\n{json.dumps(verification or {}, ensure_ascii=False)}"
    )
    return _call(api_key, COACH_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=6000)


def _fallback_narrative(profile: dict) -> str:
    sk = [(v["score"], v["label"], v["detail"], v["confidence"]) for v in profile["skills"].values()
          if v["score"] is not None]
    sk.sort()
    lines = ["**Güçlü yönler**"]
    lines += [f"- {l}: {s:.0f}/100 ({d}; güven: {c})" for s, l, d, c in sk[-3:][::-1]]
    lines += ["", "**Gelişime açık alanlar**"]
    lines += [f"- {l}: {s:.0f}/100 ({d}; güven: {c})" for s, l, d, c in sk[:3]]
    if profile["patterns"]:
        lines += ["", "**Tekrarlayan stratejik örüntüler (hipotez)**"]
        for p in profile["patterns"][:5]:
            base = f" (genel ortalama {p['baseline']:.0%})" if p["baseline"] else ""
            lines.append(f"- {p['label']}: hata oranı {p['error_rate']:.0%}{base}, {p['n_error_games']} farklı oyunda")
    lines += ["", "_Claude API anahtarı girilmediği için bu özet kurallarla üretildi._"]
    return "\n".join(lines)


# ---------------- Socratic coaching on a position ----------------
def position_context(ex: dict, engine_lines: list[dict]) -> str:
    tags = ", ".join(TAG_LABELS.get(t, t) for t in ex.get("tags", []))
    lines = "\n".join(f"  {i + 1}. {l['line']}  (eval {l['eval'] / 100:+.2f} for side to move)"
                      for i, l in enumerate(engine_lines))
    ctx = (f"Position (FEN): {ex['fen']}\nPlayer is: {'White' if ' w ' in ex['fen'] else 'Black'} (to move)\n"
           f"Context tags: {tags or '-'}\nEngine candidate lines:\n{lines}")
    if ex.get("played"):
        ctx += (f"\nIn the actual game the player played {ex['played']} (lost {ex['loss'] / 10:.0f}% of expected score); "
                f"engine best was {ex['best']}; opponent's best reply after the game move: {ex.get('refutation') or '-'}.")
    if ex.get("pattern_label"):
        ctx += f"\nThis position belongs to the hypothesis: '{ex['pattern_label']}'."
    return ctx


def coach_reply(history: list[dict], ex: dict, engine_lines: list[dict], profile: dict | None,
                api_key: str | None) -> str:
    if not available(api_key):
        return _fallback_coach(history, ex, engine_lines)
    system = COACH_SYSTEM + "\n\nPlayer profile:\n" + json.dumps(_profile_brief(profile), ensure_ascii=False) \
        if profile else COACH_SYSTEM
    system += "\n\nCurrent position:\n" + position_context(ex, engine_lines)
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    if not msgs:
        msgs = [{"role": "user", "content": "Bu pozisyonu birlikte inceleyelim. Önce bana sorularını sor."}]
    return _call(api_key, system, msgs, max_tokens=3000)


def _fallback_coach(history, ex, engine_lines) -> str:
    if not history:
        return ("Bu pozisyonda önce şu üç soruyu düşün:\n"
                "1. Rakibin bir sonraki hamlede ne yapmak istiyor? Hangi taşı ya da kareyi hedefliyor?\n"
                "2. Piyon yapısı hangi kanatta oynamanı öneriyor?\n"
                "3. En kötü durumdaki taşın hangisi ve onu nasıl iyileştirirsin?\n\n"
                "Cevabını yaz, sonra motorun değerlendirmesini birlikte karşılaştıralım.")
    best = engine_lines[0] if engine_lines else None
    txt = "Motorun önerdiği plan: " + (f"**{best['line']}** (değerlendirme {best['eval'] / 100:+.2f})." if best else "-")
    if len(engine_lines) > 1:
        txt += "\nAlternatifler: " + "; ".join(l["line"] for l in engine_lines[1:])
    txt += "\n\nKendi planınla karşılaştır: aynı hedefe mi yöneliyor? (Daha derin sohbet için Claude API anahtarı gir.)"
    return txt


# ---------------- Exercises ----------------
QUESTION_TEMPLATES = {
    "threat_awareness": "Rakibin bu pozisyondaki tehdidi ya da planı ne? Sen nasıl karşılık vermelisin?",
    "strategic_planning": "Bu pozisyonda önümüzdeki 5-6 hamle için planın ne olmalı? Hamleni ve planını yaz.",
    "positional_evaluation": "Pozisyonu değerlendir: kim daha iyi ve neden? Sonra hamleni seç.",
    "pawn_structure": "Piyon yapısı burada neyi gerektiriyor? Hangi piyon hamlesi (ya da hamlesizliği) doğru ve neden?",
    "piece_coordination": "En kötü durumdaki taşın hangisi? Taşlarının uyumunu nasıl artırırsın?",
    "advantage_conversion": "Avantajlısın. Avantajı en güvenli şekilde nasıl kazanca çevirirsin? Rakibin karşı şansı ne?",
    "defense": "Daha kötü durumdasın. Rakibin planı ne ve en dirençli savunma nedir?",
    "endgame": "Bu oyunsonunda doğru plan ne? Kral ve piyonların rolü ne olmalı?",
    "opening": "Açılışta bu pozisyonun gereği ne? Gelişim, merkez ve kral güvenliği açısından hamleni seç.",
}


def make_question(ex: dict, api_key: str | None) -> str:
    base = QUESTION_TEMPLATES.get(ex["skill"], QUESTION_TEMPLATES["strategic_planning"])
    if not available(api_key):
        return base
    prompt = ("Write ONE short training question in Turkish for this position (max 2 sentences). "
              "It must make the player think about plans and the opponent's ideas, must not reveal the answer "
              "or name any move. Return only the question.\n\n"
              f"Skill being trained: {ex['skill']}\nTemplate idea: {base}\n" + position_context(ex, ex.get("lines", [])))
    try:
        return _call(api_key, COACH_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=1500, effort="low").strip()
    except RuntimeError:
        return base


GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "move_score": {"type": "number", "description": "0..1 quality of the chosen move"},
        "reasoning_score": {"type": "number", "description": "0..1 quality of the explanation/plan"},
        "understood_opponent_plan": {"type": "boolean"},
        "feedback": {"type": "string", "description": "Turkish coaching feedback, markdown, <= 150 words"},
        "follow_up_question": {"type": "string", "description": "one Turkish follow-up question"},
        "thinking_insight": {"type": "string", "description": "what this answer suggests about how the player thinks"},
    },
    "required": ["move_score", "reasoning_score", "understood_opponent_plan", "feedback",
                 "follow_up_question", "thinking_insight"],
    "additionalProperties": False,
}


def move_score(move_eval: int | None, best_eval: int) -> float:
    """1.0 for the best move, 0 for a move losing ≥ 200 cp versus best."""
    if move_eval is None:
        return 0.0
    return round(max(0.0, min(1.0, 1 - (best_eval - move_eval) / 200)), 2)


def grade_answer(ex: dict, user_move: str, move_eval: int | None, user_text: str, api_key: str | None) -> dict:
    best_eval = ex["lines"][0]["eval"] if ex.get("lines") else 0
    ms = move_score(move_eval, best_eval)
    if not available(api_key) or not user_text.strip():
        best = ex["lines"][0] if ex.get("lines") else {"line": ex.get("best", ""), "eval": 0}
        fb = (f"Hamlen **{user_move or '-'}**: puan {ms:.0%}. Motorun planı: **{best['line']}**.\n\n"
              "Kendi açıklamanı bu planla karşılaştır: rakibin fikrini doğru gördün mü?")
        return {"score": ms, "move_score": ms, "reasoning_score": None, "feedback": fb,
                "follow_up_question": "", "thinking_insight": "", "understood_opponent_plan": None}
    prompt = ("Grade the player's answer to a training exercise. The move score from the engine is given; "
              "judge the explanation (plan, evaluation, opponent's ideas) against the engine lines. "
              "Be fair to correct ideas expressed informally.\n\n"
              f"Question: {ex['question']}\n" + position_context(ex, ex["lines"]) +
              f"\n\nPlayer's move: {user_move or '(none)'} (engine move score {ms})\nPlayer's explanation: {user_text}")
    data = json.loads(_call(api_key, COACH_SYSTEM, [{"role": "user", "content": prompt}],
                            max_tokens=4000, schema=GRADE_SCHEMA))
    rs = max(0.0, min(1.0, float(data["reasoning_score"])))
    data["move_score"] = ms
    data["reasoning_score"] = rs
    data["score"] = round(0.5 * ms + 0.5 * rs, 2)
    return data

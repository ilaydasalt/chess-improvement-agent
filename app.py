from pathlib import Path

import chess
import chess.svg
import pandas as pd
import streamlit as st

from src import coach, store
from src.analyzer import engine_lines, open_engine
from src.features import TAG_LABELS
from src.pipeline import update_player
from src.profile import SKILLS, SKILL_HELP, verification
from src.training import evaluate_move, generate_exercises

GAME_COUNT = 100  # newest Rapid games used for the profile

st.set_page_config(page_title="AI Chess Coach", page_icon="♟", layout="wide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    username = st.text_input("Chess.com username", value="saltm8").strip()
    _local_sf = Path(__file__).resolve().parents[2] / "stockfish" / "stockfish-windows-x86-64-universal.exe"
    engine_path = st.text_input("Stockfish executable path",
                                value=str(_local_sf) if _local_sf.exists() else "stockfish")
    depth = st.slider("Analysis depth", 8, 18, 12,
                      help="Depth for analysing games. Already analysed games are reused.")
    api_key = st.text_input("Claude API key (optional)", type="password",
                            help="Enables the conversational coach, written profile and graded explanations. "
                                 "Without it the app uses rule-based text. Also read from ANTHROPIC_API_KEY.") or None
    st.caption("Coach: " + ("Claude (claude-opus-5-5)" if coach.available(api_key) else "rule-based fallback"))

st.title("♟ AI Chess Coach")
st.caption(f"Learns your strategic habits from your newest {GAME_COUNT} Rapid games, "
           "tests them with exercises and tracks your progress.")

if not username:
    st.info("Enter your Chess.com username in the sidebar.")
    st.stop()

ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("coach_pos", None)
ss.setdefault("narratives", {})
ss.setdefault("current_ex", None)
ss.setdefault("last_grade", None)

profiles = store.load_profiles(username)
profile = profiles[-1] if profiles else None
attempts = store.load_attempts(username)


def board_svg(fen: str, size: int = 360, arrow: str | None = None) -> str:
    board = chess.Board(fen)
    arrows = []
    if arrow:
        try:
            m = board.parse_san(arrow)
            arrows = [chess.svg.Arrow(m.from_square, m.to_square, color="#d9534f")]
        except ValueError:
            pass
    return chess.svg.board(board, size=size, orientation=board.turn, arrows=arrows)


@st.cache_data(max_entries=200, show_spinner="Engine is thinking…")
def cached_lines(fen: str, path: str, d: int = 16) -> list[dict]:
    eng = open_engine(path)
    try:
        return engine_lines(eng, fen, depth=d, multipv=3)
    finally:
        eng.quit()


def open_in_coach(pos: dict):
    ss.coach_pos = pos
    ss.chat = []


section = st.segmented_control(
    "Section", ["Profile", "Patterns", "Coach", "Training", "Progress"],
    default="Profile", label_visibility="collapsed", key="section")

# ---------------- Profile ----------------
if section == "Profile":
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.subheader("Player profile")
    if c2.button(":material/refresh: Update profile", type="primary", width="stretch",
                 help=f"Fetch your newest {GAME_COUNT} Rapid games, analyse new ones, save a profile snapshot."):
        bar = st.progress(0.0, text="Starting…")
        try:
            profile = update_player(username, engine_path, depth=depth, count=GAME_COUNT,
                                    progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
            st.rerun()
        except FileNotFoundError:
            st.error("Stockfish was not found. Check the path in the sidebar.")
        except Exception as exc:
            st.error(f"Update failed: {type(exc).__name__}: {exc}")

    if not profile:
        st.info(f"No profile yet. Click **Update profile** to analyse your newest {GAME_COUNT} Rapid games "
                "(first run takes a while; later runs only analyse new games).")
        st.stop()

    r = profile["record"]
    m = st.container(horizontal=True)
    m.metric("Games", profile["n_games"], border=True)
    m.metric("Record (W/D/L)", f"{r['wins']}/{r['draws']}/{r['losses']}", border=True)
    m.metric("Rating", profile["elo"] or "–", border=True)
    m.metric("Period", f"{profile['first_game']} → {profile['last_game']}", border=True)

    st.markdown("#### Strategic skills")
    st.caption("Engine-based estimates (0–100). 'Avg loss' = average drop in winning chances per move (×1000). "
               "Exercise evidence confirms or rejects them.")
    rows = []
    for key, v in profile["skills"].items():
        ver = verification(attempts, key, "skill")
        rows.append({"Skill": v["label"], "Score": v["score"], "Confidence": v["confidence"],
                     "Positions": v["n"], "Exercise check": ver["status"],
                     "Evidence": v["detail"], "What it measures": SKILL_HELP[key]})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, alt="Skill scores table",
                 column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})

    st.markdown("#### Coach's summary")
    pid = profile["id"]
    if pid in ss.narratives:
        st.markdown(ss.narratives[pid])
    elif st.button(":material/psychology: Write my profile summary"):
        ver = {k: verification(attempts, k, "skill") for k in SKILLS}
        with st.spinner("Coach is reading your games…"):
            try:
                ss.narratives[pid] = coach.profile_narrative(profile, api_key, ver)
            except RuntimeError as exc:
                st.error(str(exc))
        st.rerun()

# ---------------- Patterns ----------------
elif section == "Patterns":
    st.subheader("Recurring strategic patterns")
    st.caption("Contexts where your error rate is clearly above your average, across several different games. "
               "These are hypotheses: the Training section tests them.")
    if not profile:
        st.info("Build your profile first.")
        st.stop()
    if not profile["patterns"]:
        st.success("No strong recurring pattern found in these games.")
    for p in profile["patterns"]:
        ver = verification(attempts, p["id"], "pattern_id")
        base = f" vs your average {p['baseline']:.0%}" if p["baseline"] else ""
        title = f"{p['label']} — error rate {p['error_rate']:.0%}{base} · {p['n_error_games']} games · {ver['status']}"
        with st.expander(title):
            st.write(f"Positions: {p['n_positions']} in {p['n_games']} games. Dominant cause: **{p['dominant_error']}**. "
                     f"Cause breakdown: {p['error_types'] or '–'}")
            for i, ex in enumerate(p["examples"][:3]):
                c1, c2 = st.columns([1, 2])
                c1.image(board_svg(ex["fen"], 260, ex["played"]), alt="Position before the game move")
                c2.markdown(f"**{ex['game_date']}** · you played **{ex['played']}** (−{ex['loss'] / 10:.0f}% win chance), "
                            f"engine preferred **{ex['best']}**.  \n"
                            f"Context: {', '.join(TAG_LABELS.get(t, t) for t in ex['tags'])}")
                if ex["game_url"].startswith("http"):
                    c2.link_button("Open game", ex["game_url"])
                if c2.button("Discuss with coach", key=f"disc_{p['id']}_{i}"):
                    open_in_coach({**ex, "pattern_label": p["label"]})
                    st.toast("Position sent to the Coach section.")

# ---------------- Coach ----------------
elif section == "Coach":
    st.subheader("Think it through with your coach")
    with st.expander("Choose a position", expanded=ss.coach_pos is None):
        fen = st.text_input("FEN", placeholder="Paste a FEN, or use 'Discuss with coach' from Patterns/Training")
        if st.button("Use this position") and fen:
            try:
                chess.Board(fen)
                open_in_coach({"fen": fen, "tags": []})
            except ValueError:
                st.error("Invalid FEN.")
    pos = ss.coach_pos
    if not pos:
        st.info("Pick a position to start.")
        st.stop()

    lines = cached_lines(pos["fen"], engine_path)
    left, right = st.columns([1, 1.4])
    with left:
        st.image(board_svg(pos["fen"], 380), alt="Coaching position")
        st.caption(f"{'White' if ' w ' in pos['fen'] else 'Black'} to move")
        with st.popover("Engine lines (spoiler)"):
            for l in lines:
                st.write(f"{l['eval'] / 100:+.2f}  {l['line']}")
    with right:
        chat = st.container(height=480)
        if not ss.chat:
            with st.spinner("Coach is looking at the position…"):
                try:
                    ss.chat.append({"role": "assistant",
                                    "content": coach.coach_reply([], pos, lines, profile, api_key)})
                except RuntimeError as exc:
                    st.error(str(exc))
        for msg in ss.chat:
            chat.chat_message(msg["role"]).markdown(msg["content"])
        if prompt := st.chat_input("Your thoughts, plan or question…"):
            ss.chat.append({"role": "user", "content": prompt})
            chat.chat_message("user").markdown(prompt)
            history = ss.chat
            if history and history[0]["role"] == "assistant":  # API expects a user turn first
                history = [{"role": "user", "content": "Bu pozisyonu birlikte inceleyelim."}] + history
            with chat.chat_message("assistant"), st.spinner("Thinking…"):
                try:
                    reply = coach.coach_reply(history, pos, lines, profile, api_key)
                except RuntimeError as exc:
                    reply = f"_{exc}_"
                st.markdown(reply)
            ss.chat.append({"role": "assistant", "content": reply})

# ---------------- Training ----------------
elif section == "Training":
    st.subheader("Personal training")
    st.caption("Positions from your own games, chosen from your patterns and weakest skills. "
               "One of each set is a control position where you played well.")
    if not profile:
        st.info("Build your profile first.")
        st.stop()

    exercises = store.load_exercises(username)
    done_ids = {a["exercise_id"] for a in attempts}
    open_ex = [e for e in exercises if e["id"] not in done_ids]

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.write(f"Open exercises: **{len(open_ex)}** · completed: **{len(done_ids)}**")
    if c2.button(":material/add: New set of 5", width="stretch"):
        with st.spinner("Selecting positions and preparing questions…"):
            games = store.load_games(username, profile.get("game_urls"))
            try:
                generate_exercises(username, profile, games, engine_path, api_key, n=5)
            except Exception as exc:
                st.error(f"Could not create exercises: {exc}")
        ss.last_grade = None
        st.rerun()

    if ss.last_grade:
        g, ex = ss.last_grade
        with st.container(border=True):
            st.markdown(f"**Result: {g['score']:.0%}** (move {g['move_score']:.0%}"
                        + (f", reasoning {g['reasoning_score']:.0%})" if g.get("reasoning_score") is not None else ")"))
            st.markdown(g["feedback"])
            if g.get("follow_up_question"):
                st.markdown(f"**Next question:** {g['follow_up_question']}")
            st.caption(f"In your game you played {ex['played']}; engine: {ex['lines'][0]['line']}."
                       + (" This was a control position (you played well in the game)." if ex["control"] else ""))
            if st.button("Discuss this position with the coach"):
                open_in_coach(ex)
                st.toast("Position sent to the Coach section.")

    if not open_ex:
        st.info("No open exercises. Create a new set.")
    else:
        ex = open_ex[-1]  # oldest open first
        with st.container(border=True):
            left, right = st.columns([1, 1.3])
            left.image(board_svg(ex["fen"], 360), alt="Exercise position")
            with right:
                st.markdown(f"**{SKILLS.get(ex['skill'], ex['skill'])}** · "
                            f"{'White' if ' w ' in ex['fen'] else 'Black'} to move")
                st.markdown(f"### {ex['question']}")
                with st.form(f"ex_{ex['id']}", border=False):
                    mv = st.text_input("Your move (SAN, e.g. Nf3, exd5, O-O)")
                    why = st.text_area("Your reasoning: opponent's plan, your plan, why this move",
                                       height=140)
                    sent = st.form_submit_button("Submit answer", type="primary")
                if sent:
                    norm, ev = evaluate_move(engine_path, ex["fen"], mv) if mv.strip() else (None, None)
                    if mv.strip() and norm is None:
                        st.error("That move is not legal in this position.")
                    else:
                        with st.spinner("Evaluating your answer…"):
                            try:
                                g = coach.grade_answer(ex, norm or "", ev, why, api_key)
                            except (RuntimeError, ValueError) as exc:
                                st.error(str(exc))
                                st.stop()
                        store.save_attempt(username, ex, g["score"],
                                           {**g, "answer_move": norm, "answer_text": why})
                        ss.last_grade = (g, ex)
                        st.rerun()

# ---------------- Progress ----------------
elif section == "Progress":
    st.subheader("Progress")
    if len(profiles) == 0:
        st.info("Build your profile first.")
        st.stop()
    first, last = profiles[0], profiles[-1]
    st.caption(f"First profile: {first['first_game']} → {first['last_game']} · "
               f"latest: {last['first_game']} → {last['last_game']} · {len(profiles)} snapshots")
    rows = []
    for key, label in SKILLS.items():
        a, b = first["skills"][key]["score"], last["skills"][key]["score"]
        ver = verification(attempts, key, "skill")
        rows.append({"Skill": label, "First": a, "Latest": b,
                     "Change": (b - a) if a is not None and b is not None else None,
                     "Exercises": ver["n"], "Exercise avg": ver["avg"], "Check": ver["status"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Skill change table",
                 column_config={"Change": st.column_config.NumberColumn(format="%+.1f")})
    if len(profiles) == 1:
        st.info("Progress appears after your next profile update with new games.")
    else:
        hist = pd.DataFrame([{"snapshot": p["last_game"] + f" #{p['id']}",
                              **{SKILLS[k]: p["skills"][k]["score"] for k in SKILLS}} for p in profiles])
        st.line_chart(hist.set_index("snapshot"), alt="Skill scores per profile snapshot")

    if attempts:
        st.markdown("#### Exercise results")
        adf = pd.DataFrame(attempts)
        adf["date"] = pd.to_datetime(adf["created_at"]).dt.date
        adf["skill"] = adf["skill"].map(SKILLS)
        st.bar_chart(adf.groupby("skill")["score"].mean(), horizontal=True,
                     alt="Average exercise score per skill")
        if adf["thinking_insight"].fillna("").str.len().sum() if "thinking_insight" in adf else False:
            st.markdown("#### What your answers reveal")
            for _, a in adf[adf["thinking_insight"].fillna("") != ""].tail(5).iterrows():
                st.write(f"- {a['date']} · {a['skill']}: {a['thinking_insight']}")

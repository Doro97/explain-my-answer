"""
Prototype: verification layer for "Explain My Answer" (exam-bound Swahili learner).

What is real:  inputs, navigation, state, the reference check (a rule lookup that compares
               the claim made by an explanation with a curated reference), event logging.
The AI model: explanations are generated live by Claude. The model also labels the
               grammar claim it just made, and the reference check compares that label with the
               curated reference.
Seeded error:  for exercise 2 the model is instructed to state a wrong rule (a research stimulus),
               so every tester meets one Conflict case. Without an API key, scripted text is used.
Not affiliated with Duolingo. Student prototype for testing a design hypothesis.
"""
import datetime as dt
import json
import os
import time

import anthropic
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Explain My Answer: verification prototype", layout="centered")

# --------------------------------------------------------------------------------------
# Curated reference set (illustrative prototype data, not an official source)
# Each entry: trigger word/topic -> the value a correct explanation must state.
# --------------------------------------------------------------------------------------
REFERENCES = {
    "adjective": {
        "id": "N-03",
        "value": "matches_noun_class",
        "title": "Adjective agreement with noun class",
        "rule": "An adjective takes the prefix of the class of the noun it describes: kitabu kikubwa (ki-), mtu mkubwa (m-).",
        "example": "Nilisoma kitabu kikubwa.",
    },
    "negative_past": {
        "id": "C-11",
        "value": "ku_marker",
        "title": "Negative past tense",
        "rule": "The negative past uses the negative subject prefix (si-, hu-, ha-...) + -ku- + verb stem. The marker -li- is for the POSITIVE past only.",
        "example": "Sikununua mkate jana.",
    },
}

# --------------------------------------------------------------------------------------
# Scripted exercises (the "model" output)
# --------------------------------------------------------------------------------------
EXERCISES = [
    {
        "key": "ex1_adjective_agreement",
        "prompt": "I read a big book.",
        "scripted": "Nilisoma kitabu mkubwa.",
        "is_correct": False,
        "correct": "Nilisoma kitabu kikubwa.",
        "explanation": (
            "In Swahili an adjective agrees with the noun class of its noun. **Kitabu** is in the "
            "ki-/vi- class, so the adjective takes **ki-**: *kitabu **kikubwa***. The form *mkubwa* "
            "belongs with m-/wa- nouns, for example *mtu mkubwa* (a big person)."
        ),
        "claim": ("adjective", "matches_noun_class", "The adjective must match the noun class of the noun"),
        "seed": "",
    },
    {
        "key": "ex2_negative_past_PLANTED_ERROR",
        "prompt": "I did not buy bread.",
        "scripted": "Sinunua mkate.",
        "is_correct": False,
        "correct": "Sikununua mkate.",
        "explanation": (
            "The negative past is formed with **si-** plus the past marker **-li-**, just like the "
            "positive past, so the verb becomes: *Si**li**nunua mkate.*"
        ),  # DELIBERATELY WRONG: the negative past uses -ku-, not -li-
        "claim": ("negative_past", "li_marker", "The negative past uses -li-"),
        # Research stimulus: the model is told to give this WRONG rule so testers meet a Conflict case.
        "seed": (
            "RESEARCH STIMULUS: for this exercise only, explain that the negative past is formed with "
            "the negative subject prefix plus the past marker -li- (giving 'Silinunua mkate'). "
            "Present it confidently, as a normal explanation, and set claim_value to li_marker."
        ),
    },
    {
        "key": "ex3_register",
        "prompt": "Yesterday we went to the cinema.",
        "scripted": "Jana tulienda sinema.",
        "is_correct": True,
        "correct": "Jana tulienda sinema.",
        "explanation": (
            "Your sentence is grammatically correct. In everyday conversation, Swahili speakers often "
            "add the locative *kwenye*: *Jana tulienda kwenye sinema.*"
        ),
        "claim": ("locative_in_speech", "kwenye_in_speech", "Speakers often add kwenye in conversation"),
        "seed": "",
    },
]
N = len(EXERCISES)

MODEL = "claude-sonnet-5-5"
SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string", "description": "2-4 sentences, English, markdown; Swahili in *italics*, key forms in **bold**."},
        "claim_topic": {"type": "string", "enum": ["adjective", "negative_past", "other"]},
        "claim_value": {"type": "string", "enum": ["matches_noun_class", "ku_marker", "li_marker", "other"]},
        "claim_summary": {"type": "string", "description": "The main rule your explanation asserts, in one short sentence."},
    },
    "required": ["explanation", "claim_topic", "claim_value", "claim_summary"],
    "additionalProperties": False,
}
SYSTEM = (
    "You are the 'Explain My Answer' feature of a language-learning app. The learner is practising "
    "Swahili for an exam and has just translated an English sentence. Explain why their answer is "
    "right or wrong and give the correct form. Be concise. Then label the single main grammar claim your "
    "explanation makes: claim_topic is 'adjective' (adjective agreement with noun class), 'negative_past' "
    "(forming the negative past tense) or 'other'; claim_value is 'matches_noun_class' (adjective takes the "
    "noun's class prefix), 'ku_marker' (negative past uses -ku-), 'li_marker' (negative past uses -li-) or "
    "'other'. Label what your explanation actually says."
)


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def generate_explanation(ex, typed):
    """Ask Claude to explain the learner's answer. Falls back to the scripted text on any failure."""
    scripted = dict(source="scripted", explanation=ex["explanation"], claim=ex["claim"], error="")
    key = get_api_key()
    if not key:
        scripted["error"] = "no ANTHROPIC_API_KEY"
        return scripted
    system = SYSTEM + ("\n\n" + ex["seed"] if ex["seed"] else "")
    user = (
        f"English sentence: {ex['prompt']}\n"
        f"Learner's Swahili answer: {typed}\n"
        f"Reference answer: {ex['correct']}"
    )
    try:
        resp = anthropic.Anthropic(api_key=key).beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused")
        data = json.loads(next(b.text for b in resp.content if b.type == "text"))
        return dict(
            source="ai",
            explanation=data["explanation"],
            claim=(data["claim_topic"], data["claim_value"], data["claim_summary"]),
            error="",
        )
    except Exception as e:  # keep the session usable during a test
        scripted["error"] = f"{type(e).__name__}: {e}"[:300]
        return scripted


def check_claim(claim):
    """Compare the claim an explanation makes with the curated reference. Returns (status, reference|None)."""
    trigger, value, _ = claim
    ref = REFERENCES.get(trigger)
    if ref is None:
        return "unverified", None
    return ("verified" if ref["value"] == value else "conflict"), ref


STATUS_UI = {
    "verified": ("✔ Verified: matches our curated grammar reference", "success"),
    "conflict": ("⚠ Conflict: this contradicts our curated grammar reference", "error"),
    "unverified": ("ℹ Not in our reference set: we could not check this explanation", "info"),
}

# --------------------------------------------------------------------------------------
# State + logging
# --------------------------------------------------------------------------------------
ss = st.session_state
FACILITATOR = st.query_params.get("facilitator") == "1"

if "init" not in ss:
    ss.init = True
    ss.page = -1  # -1 intro, 0..N-1 exercises, N end
    ss.t0 = None
    ss.events = []
    ss.tester = ""
    ss.stage = {}
    ss.ai = {}


def log(event, detail="", ex=None):
    now = time.time()
    ss.events.append(
        {
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "t_sec": round(now - ss.t0, 1) if ss.t0 else 0.0,
            "tester": ss.tester or "unnamed",
            "exercise": EXERCISES[ex]["key"] if ex is not None else "",
            "event": event,
            "detail": detail,
        }
    )


def stage(i):
    return ss.stage.setdefault(
        i, dict(checked=False, explained=False, src=False, teacher=False, sent=False, decision=None)
    )


def start(tester):
    ss.tester = tester.strip()
    ss.t0 = time.time()
    ss.page = 0
    log("session_start")


def do_check(i):
    typed = ss.get(f"ans_{i}", "")
    stage(i)["checked"] = True
    stage(i)["typed"] = typed
    log("answer_checked", f"typed={typed!r}; scripted_match={typed.strip() == EXERCISES[i]['scripted']}", i)


def do_explain(i):
    stage(i)["explained"] = True
    log("explain_opened", "", i)


def toggle_src(i):
    stage(i)["src"] = not stage(i)["src"]
    log("source_panel_" + ("opened" if stage(i)["src"] else "closed"), "", i)


def toggle_teacher(i):
    stage(i)["teacher"] = not stage(i)["teacher"]
    log("ask_teacher_" + ("opened" if stage(i)["teacher"] else "closed"), "", i)


def send_teacher(i):
    stage(i)["sent"] = True
    log("ask_teacher_sent", f"question={ss.get(f'q_{i}', '')!r}", i)


def decide(i, choice):
    stage(i)["decision"] = choice
    log("decision", choice, i)


def go_next(i):
    if not stage(i)["explained"]:
        log("skipped_explanation", "", i)
    ss.page = i + 1
    if ss.page >= N:
        log("session_end")


def restart():
    for k in list(ss.keys()):
        del ss[k]


# --------------------------------------------------------------------------------------
# Facilitator sidebar (open the app with ?facilitator=1)
# --------------------------------------------------------------------------------------
if FACILITATOR:
    with st.sidebar:
        st.subheader("Facilitator view")
        st.caption("Live event log. Do not explain the interface to the tester; note where they pause.")
        if ss.events:
            st.dataframe(pd.DataFrame(ss.events)[["t_sec", "exercise", "event", "detail"]], hide_index=True)
        st.button("Reset session", on_click=restart)

# --------------------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------------------
st.title("Explain My Answer")
st.caption("Verification prototype · AI explanations are generated live")


def intro():
    st.markdown(
        "Imagine you are practising Swahili for your **exam**. "
        "Work through three short exercises the way you normally would. "
        "Please **think aloud** as you go. There are no right or wrong reactions."
    )
    st.text_input("Tester label (e.g. T1)", key="tester_input")
    st.button("Start", type="primary", on_click=lambda: start(ss.get("tester_input", "")))


def exercise(i):
    ex, s = EXERCISES[i], stage(i)
    st.progress(i / N)
    st.caption(f"Exercise {i + 1} of {N}")
    st.subheader("Translate this sentence into Swahili")
    st.markdown(f"> {ex['prompt']}")

    if not s["checked"]:
        st.text_input("Your answer", value=ex["scripted"], key=f"ans_{i}")
        st.caption("Your answer is pre-filled so every tester sees the same explanations. You may edit it.")
        st.button("Check", type="primary", on_click=do_check, args=(i,))
        return

    typed = s.get("typed", "").strip()
    st.markdown(f"**Your answer:** {typed}")
    if typed == ex["correct"]:
        st.success("Correct")
    else:
        st.error("Not quite")
        st.markdown(f"**Correct answer:** {ex['correct']}")

    if not s["explained"]:
        st.button("Explain My Answer", on_click=do_explain, args=(i,), key=f"exp_{i}")
        st.button("Continue without explanation", on_click=go_next, args=(i,), key=f"skip_{i}")
        return

    # --- explanation card -------------------------------------------------------------
    if i not in ss.ai:
        with st.spinner("Writing an explanation..."):
            ss.ai[i] = generate_explanation(ex, typed)
        g = ss.ai[i]
        log("explanation_generated", f"source={g['source']}; claim={g['claim'][:2]}; error={g['error']}", i)
    g = ss.ai[i]
    status, ref = check_claim(g["claim"])
    badge_key = f"badge_{i}"
    if badge_key not in ss:
        ss[badge_key] = True
        log("badge_shown", status, i)
    label, kind = STATUS_UI[status]
    if FACILITATOR and g["source"] != "ai":
        st.warning(f"Facilitator: scripted fallback used ({g['error']})")
    with st.container(border=True):
        st.markdown("**AI explanation**")
        st.markdown(g["explanation"])
        getattr(st, kind)(label)

        st.button(
            "Hide source rule" if s["src"] else "Show source rule",
            on_click=toggle_src, args=(i,), key=f"src_{i}",
        )
        if s["src"]:
            if status == "verified":
                st.markdown(f"**{ref['title']}** (reference {ref['id']})\n\n{ref['rule']}\n\n*Example:* {ref['example']}")
            elif status == "conflict":
                st.markdown(
                    f"**This explanation says:** {g['claim'][2]}.\n\n"
                    f"**Our reference ({ref['id']}, {ref['title']}) says:** {ref['rule']}\n\n"
                    f"*Example:* {ref['example']}\n\nWe recommend checking before relying on this explanation."
                )
            else:
                st.markdown(
                    "No entry in our curated grammar reference covers this claim, so we cannot confirm "
                    "or contradict it. You may want to ask a teacher."
                )

        st.button(
            "Cancel" if s["teacher"] else "Ask a teacher",
            on_click=toggle_teacher, args=(i,), key=f"tch_{i}",
        )
        if s["teacher"] and not s["sent"]:
            st.text_area("Your question", key=f"q_{i}", placeholder="What would you like checked?")
            st.button("Send", on_click=send_teacher, args=(i,), key=f"snd_{i}")
        if s["sent"]:
            st.success("Question queued. In the real product a volunteer teacher would reply within 24 hours. "
                       "(Prototype: no message is actually sent.)")

    if s["decision"] is None:
        st.markdown("**What will you do with this explanation?**")
        c1, c2 = st.columns(2)
        c1.button("Got it: I'll use this", on_click=decide, args=(i, "accepted"), key=f"acc_{i}")
        c2.button("This looks wrong to me", on_click=decide, args=(i, "rejected"), key=f"rej_{i}")
    else:
        st.caption(f"Your choice: {'used the explanation' if s['decision'] == 'accepted' else 'marked as wrong'}")
        st.button("Continue", type="primary", on_click=go_next, args=(i,), key=f"next_{i}")


def end():
    st.progress(1.0)
    st.subheader("Thank you")
    st.markdown(
        "You've finished the exercises. **One explanation in this test was deliberately wrong** "
        "(exercise 2: the negative past uses *-ku-*, not *-li-*). One explanation was seeded with an error on purpose."
    )
    st.markdown("Please tell the facilitator what you noticed and what the three labels meant to you.")
    df = pd.DataFrame(ss.events)
    st.download_button(
        "Download session log (CSV)", df.to_csv(index=False).encode(), file_name=f"session_{ss.tester or 'unnamed'}.csv",
        mime="text/csv",
    )
    if FACILITATOR:
        st.dataframe(df, hide_index=True)
    st.button("Start over", on_click=restart)


if ss.page == -1:
    intro()
elif ss.page < N:
    exercise(ss.page)
else:
    end()
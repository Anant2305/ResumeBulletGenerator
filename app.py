from __future__ import annotations

import html

import streamlit as st
from dotenv import load_dotenv

from generator import Bullet, bullets_to_markdown, generate_bullets

load_dotenv()

st.set_page_config(
    page_title="Resume Bullet Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1100px; }
      .hero { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0.2rem; }
      .sub { color: #5b6472; margin-bottom: 1.2rem; }
      .bullet-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.1rem 0.85rem;
        margin-bottom: 0.75rem;
      }
      .star-label { font-size: 0.72rem; font-weight: 700; color: #4338ca; text-transform: uppercase; letter-spacing: 0.04em; }
      .hint { color: #64748b; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {"bullets": [], "error": None, "used_fallback": False}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_bullet(index: int, bullet: Bullet) -> None:
    st.markdown(
        f'<div class="bullet-card"><strong>{index}.</strong> {html.escape(bullet.text)}</div>',
        unsafe_allow_html=True,
    )
    st.code(bullet.text, language=None)
    if any([bullet.situation, bullet.task, bullet.action, bullet.result]):
        with st.expander("STAR breakdown"):
            cols = st.columns(4)
            pairs = [
                ("Situation", bullet.situation),
                ("Task", bullet.task),
                ("Action", bullet.action),
                ("Result", bullet.result),
            ]
            for col, (label, value) in zip(cols, pairs):
                col.markdown(f'<div class="star-label">{label}</div>', unsafe_allow_html=True)
                col.write(value or "—")
            if bullet.metric:
                st.caption(f"Metric to confirm: {bullet.metric}")


_init_state()

with st.sidebar:
    st.header("Generator")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Optional. Without a key, the app still drafts 5 STAR bullets from your words and leaves metrics as placeholders.",
        value="",
    )
    model = st.text_input("Model", value="gpt-4o-mini")
    seniority = st.selectbox(
        "Seniority voice",
        options=["junior", "mid", "senior", "staff", "lead"],
        index=1,
        help="Changes verbs and ownership language, not the facts you provide.",
    )
    st.caption("Replace any [bracketed] numbers with your real metrics before pasting onto a resume.")

st.markdown('<div class="hero">Resume bullet generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Enter the role, the impact, and the hard part. Get 5 quantified STAR bullets you can paste onto a resume.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("Inputs")
    role = st.text_input(
        "Job role",
        placeholder="e.g. Senior Backend Engineer, Product Manager, Data Analyst",
    )
    impact = st.text_area(
        "Impact created",
        height=140,
        placeholder="What got better because of your work? Who felt it? Any numbers you remember — users, time saved, revenue, quality, cost.",
    )
    challenges = st.text_area(
        "Challenges faced",
        height=140,
        placeholder="What made this hard? Constraints, incidents, politics, scale, legacy systems, missing data, tight deadlines.",
    )

    generate = st.button("Generate 5 bullets", type="primary", use_container_width=True)
    if generate:
        try:
            with st.spinner("Writing STAR bullets..."):
                bullets = generate_bullets(
                    role=role,
                    impact=impact,
                    challenges=challenges,
                    api_key=api_key.strip() or None,
                    model=model.strip() or "gpt-4o-mini",
                    seniority=seniority,
                )
            st.session_state.bullets = bullets
            st.session_state.error = None
            st.session_state.used_fallback = not bool(api_key.strip())
        except Exception as exc:
            st.session_state.bullets = []
            st.session_state.error = str(exc)

with right:
    st.subheader("Resume bullets")
    if st.session_state.error:
        st.error(st.session_state.error)
    elif not st.session_state.bullets:
        st.info("Fill in the three fields and generate. Each bullet will pack Situation, Task, Action, and Result into one line.")
        st.markdown(
            '<p class="hint">Good input sounds like: “Reduced checkout failures during a Black Friday traffic spike after the payments vendor timed out.”</p>',
            unsafe_allow_html=True,
        )
    else:
        if st.session_state.used_fallback:
            st.warning("No API key — drafting from templates. Add a key for tighter, more varied bullets.")
        for i, bullet in enumerate(st.session_state.bullets, start=1):
            _render_bullet(i, bullet)

        copy_text = bullets_to_markdown(st.session_state.bullets)
        st.download_button(
            "Download bullets",
            data=copy_text,
            file_name="resume-bullets.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.text_area("Copy all", value=copy_text, height=180, label_visibility="collapsed")

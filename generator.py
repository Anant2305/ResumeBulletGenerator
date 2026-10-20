from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


STAR_SYSTEM_PROMPT = """You write resume bullet points for software and professional roles.

Return JSON only:
{"bullets":[{"text":"...","situation":"...","task":"...","action":"...","result":"...","metric":"..."}]}

Rules for each bullet:
- One line, 22–38 words, past tense, starts with a strong verb.
- Follow STAR inside that single line: context, what you owned, what you did, measurable result.
- Quantify. Prefer % , time, cost, users, revenue, error rate, cycle time, team size, SLA.
- If the user did not give a number, use a realistic placeholder like [X%] or [N users] that they can replace. Never invent fake precise numbers.
- No first person. No fluff. No buzzword stuffing. No "responsible for".
- Vary the opening verb across the 5 bullets.
- Ground every bullet in the given role, impact, and challenges. Do not invent employers or tools the user did not mention.
- Exactly 5 bullets.
"""


@dataclass(frozen=True)
class Bullet:
    text: str
    situation: str
    task: str
    action: str
    result: str
    metric: str


def generate_bullets(
    role: str,
    impact: str,
    challenges: str,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    seniority: str = "mid",
    base_url: str | None = None,
) -> list[Bullet]:
    role, impact, challenges = role.strip(), impact.strip(), challenges.strip()
    if not role or not impact or not challenges:
        raise ValueError("Role, impact, and challenges are all required.")

    key = api_key or os.getenv("OPENAI_API_KEY")
    if key:
        return _llm_bullets(
            role=role,
            impact=impact,
            challenges=challenges,
            api_key=key,
            model=model,
            seniority=seniority,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
    return _template_bullets(role, impact, challenges, seniority)


def bullets_to_markdown(bullets: list[Bullet]) -> str:
    return "\n".join(f"• {b.text}" for b in bullets)


def _llm_bullets(
    role: str,
    impact: str,
    challenges: str,
    api_key: str,
    model: str,
    seniority: str,
    base_url: str | None,
) -> list[Bullet]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    user_prompt = (
        f"Job role: {role}\n"
        f"Seniority target: {seniority}\n"
        f"Impact created:\n{impact}\n\n"
        f"Challenges faced:\n{challenges}\n"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": STAR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    return _parse_bullets(content)


def _parse_bullets(content: str) -> list[Bullet]:
    data = json.loads(_extract_json(content))
    raw = data.get("bullets") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise ValueError("The model did not return any bullets. Try again.")

    bullets: list[Bullet] = []
    for item in raw[:5]:
        if isinstance(item, str):
            bullets.append(
                Bullet(text=_clean_bullet(item), situation="", task="", action="", result="", metric="")
            )
            continue
        text = _clean_bullet(str(item.get("text") or item.get("bullet") or ""))
        if not text:
            continue
        bullets.append(
            Bullet(
                text=text,
                situation=str(item.get("situation") or "").strip(),
                task=str(item.get("task") or "").strip(),
                action=str(item.get("action") or "").strip(),
                result=str(item.get("result") or "").strip(),
                metric=str(item.get("metric") or "").strip(),
            )
        )

    if len(bullets) < 5:
        raise ValueError("Could not parse 5 resume bullets. Generate again.")
    return bullets[:5]


def _extract_json(content: str) -> str:
    content = content.strip()
    if content.startswith("{") or content.startswith("["):
        return content
    match = re.search(r"(\{.*\}|\[.*\])", content, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse model output as JSON.")
    return match.group(1)


def _clean_bullet(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\-\*\u2022]+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    if text and not text.endswith((".", "%")):
        text += "."
    return text


def _template_bullets(role: str, impact: str, challenges: str, seniority: str) -> list[Bullet]:
    """Fallback when no API key is set. Uses the user's words; does not invent metrics."""
    impact_line = _first_sentence(impact)
    challenge_line = _first_sentence(challenges)
    scope = "Led" if seniority in {"senior", "staff", "lead"} else "Drove"
    owned = "Owned" if seniority in {"senior", "staff", "lead"} else "Executed"

    drafts = [
        (
            f"{scope} {role.lower()} work through {challenge_line.lower()}, delivering {impact_line.lower()} "
            f"with measurable [X%] improvement.",
            challenge_line,
            f"Deliver outcomes as {role}",
            f"Worked through {challenge_line.lower()}",
            impact_line,
            "[X%]",
        ),
        (
            f"{owned} a focused plan against {challenge_line.lower()} and shipped {impact_line.lower()}, "
            f"reducing time-to-result by [N days/weeks].",
            challenge_line,
            "Remove the main blocker",
            "Shipped a focused plan",
            impact_line,
            "[N days/weeks]",
        ),
        (
            f"Partnered across teams as {role} to convert {challenge_line.lower()} into a repeatable workflow, "
            f"lifting [throughput/quality] by [X%].",
            challenge_line,
            "Make the work repeatable",
            "Built a cross-team workflow",
            impact_line,
            "[X%]",
        ),
        (
            f"Diagnosed {challenge_line.lower()} and implemented a targeted fix that produced {impact_line.lower()}, "
            f"cutting errors/rework by [X%].",
            challenge_line,
            "Fix the root cause",
            "Implemented a targeted fix",
            impact_line,
            "[X%]",
        ),
        (
            f"Translated {challenge_line.lower()} into prioritized delivery as {role}, resulting in {impact_line.lower()} "
            f"for [N users/customers/stakeholders].",
            challenge_line,
            "Prioritize and deliver",
            "Sequenced work to the highest-leverage outcome",
            impact_line,
            "[N users/customers]",
        ),
    ]
    return [
        Bullet(
            text=_clean_bullet(text),
            situation=situation,
            task=task,
            action=action,
            result=result,
            metric=metric,
        )
        for text, situation, task, action, result, metric in drafts
    ]


def _first_sentence(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)
    sentence = parts[0].rstrip(".")
    if len(sentence) > 140:
        sentence = sentence[:137].rsplit(" ", 1)[0] + "..."
    return sentence

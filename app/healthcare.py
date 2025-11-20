# app/healthcare.py
"""
Generate plant health care tips using the OpenAI API.

The tips are based on:
  - plant_name (from identify.py / health.py)
  - overall status: healthy / diseased / unknown
  - likely diseases: [{name, prob}, ...] from health.assess_health
  - optional overall confidence

The output is HTML suitable for the Health page.
"""

import os
import re
from typing import List, Dict, Optional

from openai import OpenAI


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_api_key = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=_api_key) if _api_key else None


def _strip_scripts(html_text: str) -> str:
    """Remove any <script> blocks for safety."""
    return re.sub(r"(?is)<script.*?>.*?</script>", "", html_text or "")


SYSTEM = (
    "You are a careful, empathetic plant health assistant.\n"
    "You get:\n"
    " - The plant name (may be scientific or common).\n"
    " - A coarse health status: healthy / diseased / unknown.\n"
    " - 1-5 likely disease patterns inferred from leaf photos, "
    "   with approximate probabilities. These labels may come "
    "   from datasets (e.g., 'late blight', 'bacterial spot') and "
    "   should be treated as pattern names, not exact diagnoses.\n\n"
    "Your job:\n"
    " - Give safe, practical advice for a home grower.\n"
    " - Stay generic: do NOT recommend specific product brands or "
    "   restricted chemicals. You can mention general types like "
    "   'a copper-based fungicide labelled for fruit trees', but "
    "   always tell them to follow local regulations and labels.\n"
    " - If diseases are low-confidence or uncertain, say so and "
    "   focus on observation + general care.\n"
    " - Never encourage risky self-treatment of people or animals; "
    "   you only advise about plants and gardening.\n\n"
    "SPECIAL CASE – WHEN STATUS IS HEALTHY:\n"
    " - Use a warm, encouraging tone. Briefly congratulate the user "
    "   for keeping the plant healthy.\n"
    " - Emphasise ongoing prevention: ideal light, watering, soil, "
    "   drainage, temperature and humidity for this plant, as far as "
    "   you reasonably know from its name.\n"
    " - Mention typical native region / climate only in a cautious way "
    "   (for example: 'often grown in warm, temperate climates').\n"
    " - Even if disease probabilities are very low, gently remind them "
    "   of a few simple checks so the plant stays in good shape.\n"
    " - Add a caring, emotional vibe (e.g. 'your plant is lucky to have you').\n\n"
    "FORMAT REQUIREMENTS (IMPORTANT):\n"
    " - Respond in VALID HTML ONLY (no Markdown, no ``` blocks).\n"
    " - Use the following elements where appropriate: "
    "   <h3>, <p>, <ol>, <ul>, <li>, <strong>, <em>, <br>.\n"
    " - Do NOT include <html>, <head>, or <body> wrappers.\n"
    " - Structure the answer as exactly 3 sections with headings:\n"
    "   <h3>What might be happening &amp; why</h3>\n"
    "   <h3>What to do now (step-by-step)</h3>\n"
    "   <h3>Prevention &amp; precautions</h3>\n"
    " - Use bullet lists (<ul>/<ol>) for steps and tips.\n"
    " - Total length around 120-200 words.\n"
)


def _build_user_prompt(
    plant_name: Optional[str],
    status: str,
    diseases: List[Dict],
    confidence: Optional[float],
    plant_confidence: Optional[float],
) -> str:
    """
    Build a detailed user message that tells the model:
      - plant name
      - status
      - list of diseases with probabilities
      - whether disease signal is low
    """
    name_txt = plant_name or "Unknown plant"
    status_txt = (status or "unknown").lower()

    # e.g. "late blight (74%), bacterial spot (12%)"
    ds_bits = []
    probs = []
    for d in diseases[:5]:
        n = str(d.get("name", "unknown issue"))
        p = float(d.get("prob", 0.0))
        ds_bits.append(f"{n} ({round(p * 100)}%)")
        probs.append(p)
    ds_text = ", ".join(ds_bits) if ds_bits else "none – model was uncertain"

    max_disease_prob = max(probs) if probs else 0.0
    health_conf_txt = (
        f"{round(float(confidence) * 100)}%" if confidence is not None else "unknown"
    )
    plant_conf_txt = (
        f"{round(float(plant_confidence) * 100)}%" if plant_confidence is not None else "unknown"
    )

    # Extra hint for the model so it can vary text for low-disease cases
    if status_txt == "healthy" or max_disease_prob < 0.20:
        situation_hint = (
            "The plant appears healthy or only shows very mild/uncertain issues. "
            "Focus mostly on appreciation, gentle encouragement, and prevention."
        )
    else:
        situation_hint = (
            "The plant appears diseased or stressed. Focus on likely causes, "
            "practical immediate steps, and clear but calm guidance."
        )

    return (
        f"Plant: {name_txt}\n"
        f"Health status: {status_txt} "
        f"(model confidence about this status: {health_conf_txt}).\n"
        f"Confidence that the plant identification is correct: {plant_conf_txt}.\n"
        f"Likely disease patterns (approximate, from leaf images): {ds_text}.\n"
        f"Maximum disease probability: {round(max_disease_prob * 100)}%.\n"
        f"Situation hint: {situation_hint}\n\n"
        "When you write the advice, tie care suggestions to the plant name if possible "
        "(typical climate, light, soil, and watering needs), but keep things safe and "
        "generic for home growers.\n"
        "Please follow the format instructions from the system message exactly."
    )


def render_care_html(
    status: str,
    diseases: List[Dict],
    plant_name: Optional[str] = None,
    plant_confidence: Optional[float] = None,
    overall_confidence: Optional[float] = None,
) -> str:
    """
    Generate AI-based care tips as HTML.

    Parameters
    ----------
    status : str
        'healthy', 'diseased', or 'unknown' from app.health.assess_health.
    diseases : list of dict
        Each dict has at least {'name': str, 'prob': float} with pretty names.
    plant_name : optional str
        Name returned by identify.py / health.assess_health (e.g. 'Malus spp.').
    plant_confidence : optional float
        Confidence for plant identification (0..1).
    overall_confidence : optional float
        Aggregate confidence about the health status (0..1).

    Returns
    -------
    HTML string (safe to drop into your template).
    """
    # If no API key configured, keep it honest and minimal.
    if client is None:
        return (
            "<p class='muted'>Care tips are unavailable because no OpenAI API key "
            "is configured on the server.</p>"
        )

    user = _build_user_prompt(
        plant_name=plant_name,
        status=status,
        diseases=diseases,
        confidence=overall_confidence,
        plant_confidence=plant_confidence,
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.45,  # a bit more creativity for nice wording
        )
        txt = (resp.choices[0].message.content or "").strip()
        if not txt:
            raise RuntimeError("empty completion")
    except Exception:
        # If the API call fails, avoid misleading content.
        return (
            "<p class='muted'>We could not generate care tips right now. "
            "Please try again later, and consider checking with a local expert.</p>"
        )

    # IMPORTANT: do NOT escape HTML here – we *want* the tags to render.
    safe_html = _strip_scripts(txt)
    return f"<div class='care-text'>{safe_html}</div>"

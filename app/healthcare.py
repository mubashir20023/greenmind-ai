# app/healthcare.py

import os
import re
from typing import List, Dict, Optional
from openai import OpenAI

MODEL_DEFAULT = "gpt-4o-mini"


# -------------------- CLIENT --------------------
def _get_client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("[healthcare] OPENAI_API_KEY is not set")
        return None
    return OpenAI(api_key=key)


# -------------------- SAFETY --------------------
def _strip_scripts(html_text: str) -> str:
    return re.sub(r"(?is)<script.*?>.*?</script>", "", html_text or "")


# -------------------- SYSTEM PROMPT --------------------
SYSTEM = (
    "You are a careful, empathetic plant health assistant.\n"
    "You get:\n"
    " - The plant name.\n"
    " - A coarse health status: healthy / diseased / unknown.\n"
    " - Likely disease patterns with probabilities.\n\n"

    "Your job:\n"
    " - Give safe, practical advice.\n"
    " - Stay generic (no specific products).\n"
    " - If uncertain → say so.\n\n"

    "FORMAT:\n"
    " - VALID HTML ONLY\n"
    " - EXACTLY 3 sections:\n"
    "   <h3>What might be happening &amp; why</h3>\n"
    "   <h3>What to do now (step-by-step)</h3>\n"
    "   <h3>Prevention &amp; precautions</h3>\n"
)


# -------------------- PROMPT BUILDER --------------------
def _build_user_prompt(
    plant_name: Optional[str],
    status: str,
    diseases: List[Dict],
    confidence: Optional[float],
    plant_confidence: Optional[float],
) -> str:

    name_txt = plant_name or "Unknown plant"
    status_txt = (status or "unknown").lower()

    # 🔧 FIX: remove duplicates (IMPORTANT)
    unique = {}
    for d in diseases[:5]:
        name = str(d.get("name", "unknown issue"))
        prob = float(d.get("prob", 0.0))
        if name not in unique or prob > unique[name]["prob"]:
            unique[name] = {"name": name, "prob": prob}

    diseases = list(unique.values())

    ds_bits = []
    probs = []

    for d in diseases:
        n = d["name"]
        p = d["prob"]
        ds_bits.append(f"{n} ({round(p * 100)}%)")
        probs.append(p)

    ds_text = ", ".join(ds_bits) if ds_bits else "none – uncertain"

    max_prob = max(probs) if probs else 0.0

    return (
        f"Plant: {name_txt}\n"
        f"Health status: {status_txt}\n"
        f"Diseases: {ds_text}\n"
        f"Max disease probability: {round(max_prob * 100)}%\n"
    )


# -------------------- MAIN FUNCTION --------------------
def render_care_html(
    status: str,
    diseases: List[Dict],
    plant_name: Optional[str] = None,
    plant_confidence: Optional[float] = None,
    overall_confidence: Optional[float] = None,
) -> str:

    client = _get_client()

    if client is None:
        return "<p class='muted'>Care tips unavailable — no API key.</p>"

    user_prompt = _build_user_prompt(
        plant_name=plant_name,
        status=status,
        diseases=diseases,
        confidence=overall_confidence,
        plant_confidence=plant_confidence,
    )

    print(">>> CARE REQUEST")
    print("Plant:", plant_name)
    print("Status:", status)
    print("Diseases:", diseases)

    try:
        model = os.getenv("OPENAI_MODEL", MODEL_DEFAULT)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=700,
            timeout=30,
        )

        content = (response.choices[0].message.content or "").strip()

        print(">>> OPENAI RESPONSE LENGTH:", len(content))

        if not content:
            print("⚠️ EMPTY RESPONSE FROM OPENAI")
            return _fallback_tips(plant_name, status)

        safe_html = _strip_scripts(content)
        return f"<div class='care-text'>{safe_html}</div>"

    except Exception as e:
        import traceback
        traceback.print_exc()

        print("❌ OPENAI ERROR:", str(e))

        # 🔥 ALWAYS return fallback (never break UI)
        return _fallback_tips(plant_name, status)


# -------------------- FALLBACK SYSTEM --------------------
def _fallback_tips(plant_name: Optional[str], status: str) -> str:

    if status == "healthy":
        return """
        <div class='care-text'>
        <h3>What might be happening &amp; why</h3>
        <p>Your plant appears healthy and well-maintained.</p>

        <h3>What to do now (step-by-step)</h3>
        <ul>
          <li>Continue regular watering (avoid overwatering)</li>
          <li>Ensure good sunlight exposure</li>
          <li>Check leaves weekly for early signs</li>
        </ul>

        <h3>Prevention &amp; precautions</h3>
        <ul>
          <li>Maintain airflow around leaves</li>
          <li>Use clean tools</li>
          <li>Avoid water sitting on leaves</li>
        </ul>
        </div>
        """

    return """
    <div class='care-text'>
    <h3>What might be happening &amp; why</h3>
    <p>The plant may be experiencing stress or disease symptoms.</p>

    <h3>What to do now (step-by-step)</h3>
    <ul>
      <li>Remove affected leaves</li>
      <li>Avoid overwatering</li>
      <li>Improve sunlight and airflow</li>
    </ul>

    <h3>Prevention &amp; precautions</h3>
    <ul>
      <li>Keep leaves dry</li>
      <li>Inspect regularly</li>
      <li>Isolate infected plants if needed</li>
    </ul>
    </div>
    """
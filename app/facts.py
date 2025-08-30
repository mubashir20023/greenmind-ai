# app/facts.py
import os, json, logging, requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
WIKIPEDIA_TIMEOUT = int(os.getenv("WIKIPEDIA_TIMEOUT", "5"))

def get_wikipedia(label: str) -> dict:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{label.replace(' ', '_')}"
    r = requests.get(url, timeout=WIKIPEDIA_TIMEOUT)
    if r.status_code != 200:
        return {}
    j = r.json()
    return {
        "title": j.get("title"),
        "summary": j.get("extract"),
        "url": j.get("content_urls", {}).get("desktop", {}).get("page"),
        "thumbnail": j.get("thumbnail", {}).get("source"),
    }

def generate_html(label: str, confidence: float, facts_json: dict) -> str:
    # fallback if no key / API error
    if not client:
        txt = facts_json.get("summary") or f"No summary found for {label}."
        url = facts_json.get("url","")
        src = f'<p><small>Source: <a href="{url}">{url}</a></small></p>' if url else ""
        return f'<div><h3>{label}</h3><p>{txt}</p>{src}</div>'

    prompt = (
        f"Plant label/species: {label}\n"
        f"Model confidence: {confidence:.1%}\n"
        f"Facts JSON from Wikipedia:\n```json\n{json.dumps(facts_json, ensure_ascii=False, indent=2)}\n```\n"
        "Return HTML only with <div>, <h3>, <ul>, <li>, and a Sources section listing URLs. "
        "Sections: Overview, Pros/Uses, Cautions/Toxicity (if any), Fun Facts, Little-Known Facts, Sources."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a botanist and science communicator. Be accurate, concise, and safety-conscious."},
                {"role":"user","content": prompt}
            ],
            max_tokens=700,
            temperature=0.6,
        )
        html = (resp.choices[0].message.content or "").strip()
        if "<" not in html:
            html = f"<div><p>{html}</p></div>"
        return html
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"<div><h3>{label}</h3><p>{facts_json.get('summary','No summary available.')}</p></div>"

# app/facts.py
from __future__ import annotations
import os, re, json, logging, requests
from html import escape
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---- OpenAI (new SDK >=1.0) ----
try:
    from openai import OpenAI  # pip install --upgrade openai
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

WIKIPEDIA_TIMEOUT = int(os.getenv("WIKIPEDIA_TIMEOUT", "10"))
_UA = {"User-Agent": "GreenMindAI/1.0 (https://greenmind-ai.onrender.com/)"}

def _safe(s: str | None) -> str:
    return escape(s or "", quote=True)

# ---------------- Wikipedia resolution ----------------
def _wiki_summary_by_title(title: str) -> dict | None:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        r = requests.get(url, headers=_UA, timeout=WIKIPEDIA_TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            return {
                "title": j.get("title") or title.replace("_", " "),
                "summary": j.get("extract") or "",
                "url": (j.get("content_urls", {}).get("desktop", {}) or {}).get("page")
                       or f"https://en.wikipedia.org/wiki/{title}",
                "thumbnail": (j.get("thumbnail") or {}).get("source") or "",
            }
        else:
            logger.warning(f"[wiki] summary {title} -> HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"[wiki] summary failed for {title}: {e}")
    return None

def _wiki_opensearch_first(q: str) -> str | None:
    try:
        api = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}
        r = requests.get(api, params=params, headers=_UA, timeout=WIKIPEDIA_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) >= 2 and data[1]:
                return str(data[1][0])
        else:
            logger.warning(f"[wiki] opensearch {q} -> HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"[wiki] opensearch failed for {q}: {e}")
    return None

def _resolve_wikipedia(label: str) -> dict:
    variants = [
        label,
        label.replace(" ", "_"),
        label.replace("_", " "),
        label.title().replace(" ", "_"),
    ]
    for v in variants:
        page = _wiki_summary_by_title(quote(v.replace(" ", "_")))
        if page and (page.get("summary") or page.get("url")):
            return page
    best = _wiki_opensearch_first(label.replace("_", " "))
    if best:
        page = _wiki_summary_by_title(quote(best.replace(" ", "_")))
        if page and (page.get("summary") or page.get("url")):
            return page
    return {
        "title": label.replace("_", " "),
        "summary": "",
        "url": f"https://en.wikipedia.org/w/index.php?search={quote(label.replace('_', ' '))}",
        "thumbnail": "",
    }

def get_wikipedia(label: str) -> dict:
    return _resolve_wikipedia(label)

# ---------------- Heuristics for a compact non-LLM “micro-card” ----------------
_MONTHS = [
    ("january", "Jan"), ("february", "Feb"), ("march", "Mar"), ("april", "Apr"),
    ("may", "May"), ("june", "Jun"), ("july", "Jul"), ("august", "Aug"),
    ("september", "Sep"), ("october", "Oct"), ("november", "Nov"), ("december", "Dec"),
]
_M2I = {m: i for i, (m, _) in enumerate(_MONTHS)}

def _find_months_near(text: str, keyword: str) -> str:
    """Return a compact month range like 'Apr–Jun' near 'flower'/'fruit' if mentioned."""
    if not text:
        return ""
    t = text.lower()
    # Look at windows around the keyword
    out_idx = []
    for m, abbr in _MONTHS:
        # capture months within ~100 chars of keyword context
        pattern = rf"(?:{keyword}\w*[^.{{0,100}}]|[^.{{0,100}}]{keyword}\w*).*?\b{m}\b"
        if re.search(pattern, t):
            out_idx.append(_M2I[m])
        # also capture simple 'from X to Y' patterns broadly
        if re.search(rf"\b{m}\b", t) and re.search(keyword, t):
            out_idx.append(_M2I[m])
    out_idx = sorted(set(out_idx))
    if not out_idx:
        # try simple phrases: 'flowers from April to June'
        m = re.search(rf"{keyword}\w*[^.]*?\b(from|between)\s+([A-Za-z]+)\s+(?:to|and|-)\s+([A-Za-z]+)", t)
        if m:
            a, b = m.group(2).lower(), m.group(3).lower()
            if a in _M2I and b in _M2I:
                return f"{_MONTHS[_M2I[a]][1]}–{_MONTHS[_M2I[b]][1]}"
        return ""
    # compress contiguous runs into ranges
    ranges = []
    start = prev = out_idx[0]
    for i in out_idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append((start, prev))
        start = prev = i
    ranges.append((start, prev))
    parts = []
    for a, b in ranges:
        if a == b:
            parts.append(_MONTHS[a][1])
        else:
            parts.append(f"{_MONTHS[a][1]}–{_MONTHS[b][1]}")
    return ", ".join(parts)

def _heuristic_common_names(text: str) -> list[str]:
    if not text:
        return []
    lead = text.split(".")[0][:400]
    m = re.search(r"\((?:also|commonly)?\s*(?:known|called)\s*as\s*([^)]{3,120})\)", lead, flags=re.I)
    chunk = None
    if m:
        chunk = m.group(1)
    else:
        m2 = re.search(r"\(([^)]{3,120})\)", lead)
        if m2 and not re.search(r"\d", m2.group(1)):
            chunk = m2.group(1)
    if not chunk:
        return []
    parts = re.split(r"\s*(?:,|/| or )\s*", chunk, flags=re.I)
    parts = [p.strip(" .;:").lower() for p in parts if 2 < len(p) < 60]
    # remove obvious scientific markers
    parts = [p for p in parts if not re.search(r"[A-Z][a-z]+\s+[a-z]\.", p)]
    return list(dict.fromkeys(parts))[:6]

def _heuristic_where_found(text: str) -> str:
    if not text:
        return ""
    lead = text[:600]
    pats = [
        r"\bnative to ([^.;:]{3,80})", r"\bnative of ([^.;:]{3,80})",
        r"\bendemic (?:to|in) ([^.;:]{3,80})",
        r"\bfound in ([^.;:]{3,80})", r"\bdistributed in ([^.;:]{3,80})",
        r"\bwidely distributed in ([^.;:]{3,80})", r"\boccurs in ([^.;:]{3,80})",
        r"\bgrows in ([^.;:]{3,80})",
    ]
    for pat in pats:
        m = re.search(pat, lead, flags=re.I)
        if m:
            return m.groups()[-1].strip(" ,.;:")
    return ""

def _microcard_no_llm(label: str, confidence: float, facts: dict) -> str:
    title = facts.get("title") or label.replace("_", " ")
    summary = facts.get("summary") or ""
    url = facts.get("url") or f"https://en.wikipedia.org/wiki/{quote(label.replace(' ', '_'))}"

    commons = _heuristic_common_names(summary)
    where = _heuristic_where_found(summary)
    flowering = _find_months_near(summary, "flower")
    fruiting  = _find_months_near(summary, "fruit")

    bullets = []
    if commons:
        bullets.append(f"Common names: {_safe(', '.join(commons))}")
    if where:
        bullets.append(f"Where found: {_safe(where)}")
    bullets.append(f"Flowering: {_safe(flowering or 'N/A')}")
    bullets.append(f"Fruiting: {_safe(fruiting or 'N/A')}")

    parts = []
    parts.append("<div>")
    parts.append(f"<h3>{_safe(title)}</h3>")
    parts.append("<h4>Quick facts</h4>")
    parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets[:6]) + "</ul>")
    if summary:
        parts.append("<h4>Overview</h4>")
        parts.append(f"<p>{_safe(summary)}</p>")
    parts.append(f'<p><small>Model confidence: {confidence:.1%}</small></p>')
    parts.append("<h4>Sources</h4>")
    parts.append(f'<ul><li><a href="{_safe(url)}" target="_blank" rel="noreferrer">Wikipedia</a></li></ul>')
    parts.append("</div>")
    return "".join(parts)

# ---------------- LLM path: ultra-short micro-card ----------------
def generate_html(label: str, confidence: float, facts_json: dict) -> str:
    wiki = facts_json or {}
    if not wiki.get("url"):
        wiki = _resolve_wikipedia(label)

    # No LLM? Use compact heuristic micro-card
    if not client:
        return _microcard_no_llm(label, confidence, wiki)

    prompt = (
        "You are a botanist and science communicator. Write an ULTRA-CONCISE micro-card in HTML.\n\n"
        f"Plant/species: {label}\n"
        f"Model confidence: {confidence:.1%}\n"
        f"Wikipedia (resolved): {json.dumps(wiki, ensure_ascii=False)}\n\n"
        "OUTPUT RULES:\n"
        "- Use ONLY: <div>, <h3>, <h4>, <ul>, <li>, <p>, <a>.\n"
        "- Keep TOTAL under ~900 characters.\n"
        "- Use a single <ul> under a heading 'Quick facts'.\n"
        "- Bullets (only include if supported by the Wikipedia data; if unknown, omit):\n"
        "  • Common names: ...\n"
        "  • Where found: ... (distribution/regions)\n"
        "  • Flowering: months (e.g., Apr–Jun)\n"
        "  • Fruiting: months (e.g., Aug–Oct)\n"
        "  • Uses: ... (very short)\n"
        "  • Cautions: ... (toxicity, invasiveness, etc.)\n"
        "  • Fun fact: ... (1 short item)\n"
        "- After the list, add an 'Overview' section with 1–2 sentences.\n"
        "- Finish with a 'Sources' section linking the exact Wikipedia URL you were given as the anchor text 'Wikipedia'.\n"
        "- DO NOT invent data; if something isn't in the Wikipedia JSON, omit that bullet.\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be precise, concise, and never fabricate beyond the provided Wikipedia data."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.4,
        )
        html = (resp.choices[0].message.content or "").strip()
        # Ensure a Wikipedia link is present
        if "wikipedia" not in html.lower() or "<a " not in html:
            src = _safe(wiki.get("url") or f"https://en.wikipedia.org/wiki/{quote(label.replace(' ', '_'))}")
            tail = f'<h4>Sources</h4><ul><li><a href="{src}" target="_blank" rel="noreferrer">Wikipedia</a></li></ul>'
            html = (html.replace("</div>", tail + "</div>") if "</div>" in html else f"<div>{html}{tail}</div>")
        return html
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return _microcard_no_llm(label, confidence, wiki)

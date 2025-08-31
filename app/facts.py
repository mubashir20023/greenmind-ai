# app/facts.py
from __future__ import annotations
import os, re, json, logging, requests
from html import escape
from urllib.parse import quote
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Wikipedia resolution
# -----------------------------------------------------------------------------
def _wiki_summary_by_title(title: str) -> dict | None:
    """
    Resolve a page by title via REST summary endpoint.
    Returns dict with title/summary/url/thumbnail or None on failure.
    """
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
    """Use Opensearch to get the most likely page title for a query string."""
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
def _heuristic_season(text: str) -> str:
    """Extract season info from text (stub; customize as needed)."""
    # Example: look for 'season' keywords
    for s in ["spring", "summer", "autumn", "fall", "winter", "rainy season", "dry season"]:
        if s in text.lower():
            return s
    return ""
def _resolve_wikipedia(label: str) -> dict:
    """Try several title variants; fallback to a search URL if not found."""
    variants = [label, label.replace(" ", "_"), label.replace("_", " "), label.title().replace(" ", "_")]
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

def _wiki_full_extract(title: str) -> str:
    """
    Pull the full plain-text extract (all sections) for richer facts.
    Capped to avoid enormous prompts.
    """
    try:
        api = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
            "titles": title, "format": "json", "redirects": 1,
        }
        r = requests.get(api, params=params, headers=_UA, timeout=WIKIPEDIA_TIMEOUT)
        if r.status_code != 200:
            logger.warning(f"[wiki] full extract HTTP {r.status_code}")
            return ""
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        if not pages: return ""
        page = next(iter(pages.values()))
        txt = page.get("extract") or ""
        return txt[:20000]
    except Exception as e:
        logger.warning(f"[wiki] full extract failed: {e}")
        return ""

def get_wikipedia(label: str) -> dict:
    return _resolve_wikipedia(label)

# -----------------------------------------------------------------------------
# Month & season heuristics (robust: full, abbr, ranges, seasonal fallback)
# -----------------------------------------------------------------------------
_MONTHS = [
    ("january","Jan"),("february","Feb"),("march","Mar"),("april","Apr"),
    ("may","May"),("june","Jun"),("july","Jul"),("august","Aug"),
    ("september","Sep"),("october","Oct"),("november","Nov"),("december","Dec"),
]
_M2I = {m: i for i, (m, _) in enumerate(_MONTHS)}
_M3_TO_I = {"jan":0,"feb":1,"mar":2,"apr":3,"may":4,"jun":5,"jul":6,"aug":7,"sep":8,"oct":9,"nov":10,"dec":11,"sept":8}
_MONTH_TOKEN = r"(jan(?:uary)?\.?|feb(?:ruary)?\.?|mar(?:ch)?\.?|apr(?:il)?\.?|may\.?|jun(?:e)?\.?|jul(?:y)?\.?|aug(?:ust)?\.?|sep(?:t(?:ember)?)?\.?|oct(?:ober)?\.?|nov(?:ember)?\.?|dec(?:ember)?)"

def _token_to_idx(tok: str) -> int | None:
    t = tok.lower().strip(".")
    t3 = (t[:4] if t.startswith("sept") else t[:3])
    return _M3_TO_I.get(t3)

def _month_indices_in_text(text: str) -> list[int]:
    out = []
    for m in re.finditer(_MONTH_TOKEN, text, flags=re.I):
        idx = _token_to_idx(m.group(0))
        if idx is not None: out.append(idx)
    return out

def _compress_indices_to_ranges(idxs: list[int]) -> str:
    if not idxs: return ""
    idxs = sorted(set(idxs))
    ranges = []
    start = prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i; continue
        ranges.append((start, prev)); start = prev = i
    ranges.append((start, prev))
    parts = [(_MONTHS[a][1] if a == b else f"{_MONTHS[a][1]}–{_MONTHS[b][1]}") for a, b in ranges]
    return ", ".join(parts)

def _find_months_near(text: str, keyword: str, window: int = 160) -> str:
    """
    Find month tokens near a keyword ('flower'/'fruit'), including 'from X to Y' ranges.
    """
    if not text: return ""
    t = text
    idxs: list[int] = []
    # windows around keyword
    for m in re.finditer(rf"\b{keyword}\w*\b", t, flags=re.I):
        a = max(0, m.start() - window); b = min(len(t), m.end() + window)
        idxs += _month_indices_in_text(t[a:b])
        # 'from X to Y' near keyword
        near = t[a:b]
        m2 = re.search(rf"\b(from|between)\s+({_MONTH_TOKEN})\s+(?:to|and|-|–)\s+({_MONTH_TOKEN})", near, flags=re.I)
        if m2:
            i1 = _token_to_idx(m2.group(2)); i2 = _token_to_idx(m2.group(3))
            if i1 is not None and i2 is not None:
                if i1 <= i2: idxs += list(range(i1, i2+1))
                else: idxs += list(range(i2, i1+1))
    # global patterns like "flowers April–June"
    for m in re.finditer(rf"{keyword}\w*[^.]*({_MONTH_TOKEN})\s*(?:-|–|to|and)\s*({_MONTH_TOKEN})", t, flags=re.I):
        i1 = _token_to_idx(m.group(1)); i2 = _token_to_idx(m.group(2))
        if i1 is not None and i2 is not None:
            if i1 <= i2: idxs += list(range(i1, i2+1))
            else: idxs += list(range(i2, i1+1))
    return _compress_indices_to_ranges(idxs)

def _season_phrase(text: str, keyword: str) -> str:
    """If months are absent, try a seasonal phrase near the keyword."""
    if not text: return ""
    window = 160
    seasons = ["spring","summer","monsoon","rainy season","wet season","dry season","autumn","fall","winter"]
    for m in re.finditer(rf"\b{keyword}\w*\b", text, flags=re.I):
        a = max(0, m.start() - window); b = min(len(text), m.end() + window)
        chunk = text[a:b].lower()
        for s in seasons:
            if s in chunk:
                return s
    return ""

# -----------------------------------------------------------------------------
# Simple extractors (common names, where found, uses, cautions, needs, fun facts)
# -----------------------------------------------------------------------------
def _heuristic_common_names(text: str) -> list[str]:
    if not text: return []
    lead = text.split(".")[0][:700]
    m = re.search(r"\((?:also|commonly)?\s*(?:known|called)\s*as\s*([^)]{3,200})\)", lead, flags=re.I)
    chunk = m.group(1) if m else None
    if not chunk:
        m2 = re.search(r"\(([^)]{3,200})\)", lead)
        if m2 and not re.search(r"\d", m2.group(1)): chunk = m2.group(1)
    if not chunk: return []
    parts = re.split(r"\s*(?:,|/| or )\s*", chunk, flags=re.I)
    parts = [p.strip(" .;:") for p in parts if 2 < len(p) < 60]
    parts = [p for p in parts if not re.search(r"[A-Z][a-z]+\s+[a-z]\.", p)]
    parts = [p if re.search(r"[A-Z]", p) else p.title() for p in parts]
    return list(dict.fromkeys(parts))[:6]

def _heuristic_where_found(text: str) -> str:
    if not text: return ""
    lead = text[:1600]
    pats = [
        r"\bnative to ([^.;:]{3,140})", r"\bnative of ([^.;:]{3,140})",
        r"\bendemic (?:to|in) ([^.;:]{3,140})", r"\bfound in ([^.;:]{3,140})",
        r"\bdistributed in ([^.;:]{3,140})", r"\bwidely distributed in ([^.;:]{3,140})",
        r"\boccurs in ([^.;:]{3,140})", r"\bgrows in ([^.;:]{3,140})",
    ]
    for pat in pats:
        m = re.search(pat, lead, flags=re.I)
        if m: return m.groups()[-1].strip(" ,.;:")
    return ""

def _has(text: str, *words: str) -> bool:
    t = text.lower()
    return any(re.search(rf"\b{re.escape(w.lower())}\b", t) for w in words)

def _heuristic_uses(text: str) -> str:
    if not text: return ""
    tags = []
    if _has(text,"traditional medicine","ayurveda","folk medicine","medicinal"): tags.append("traditional medicine")
    if _has(text,"fiber","fibres","floss","kapok","silky floss"): tags.append("fiber/floss")
    if _has(text,"garland","leis","ornamental","hedge"): tags.append("ornamental/garlands")
    if _has(text,"dye"): tags.append("dye")
    if _has(text,"soap","detergent","saponin"): tags.append("soap/saponins")
    if _has(text,"food","edible","culinary"): tags.append("culinary")
    seen=set(); out=[]
    for t in tags:
        if t in seen: continue
        seen.add(t); out.append(t)
    return ", ".join(out)

def _heuristic_cautions(text: str) -> str:
    if not text: return ""
    tags = []
    if _has(text,"toxic","poison","poisonous","toxicity"): tags.append("toxic")
    if _has(text,"latex","milky sap","urushiol","dermatitis"): tags.append("skin/eye irritation")
    if _has(text,"invasive","weed","noxious"): tags.append("potentially invasive")
    seen=set(); out=[]
    for t in tags:
        if t in seen: continue
        seen.add(t); out.append(t)
    return ", ".join(out)

def _heuristic_growing_needs(text: str) -> str:
    if not text: return ""
    t = text.lower()
    cues = []
    if re.search(r"\b(full )?sun\b", t): cues.append("full sun")
    if re.search(r"\bpartial shade\b|\bpart(-| )shade\b", t): cues.append("part shade")
    if re.search(r"\bshade\b", t): cues.append("shade")
    if re.search(r"\bwell-?drained\b", t): cues.append("well-drained soil")
    if re.search(r"\bmoist\b", t): cues.append("moist soil")
    if re.search(r"\bdry\b", t): cues.append("tolerates dry")
    if re.search(r"\bsandy\b", t): cues.append("sandy")
    if re.search(r"\bloam\b", t): cues.append("loam")
    if re.search(r"\bcalcareous\b|\blimestone\b", t): cues.append("calcareous")
    if re.search(r"\btropical\b|\bsubtropical\b|\barid\b|\bsubalpine\b|\balpine\b", t): cues.append("climate-suited")
    if re.search(r"\bdrought-?tolerant\b", t): cues.append("drought-tolerant")
    seen=set(); out=[]
    for c in cues:
        if c in seen: continue
        seen.add(c); out.append(c)
    return ", ".join(out)

def _heuristic_fun_fact(text: str) -> str:
    """One short generic fun fact from triggers."""
    if not text: return ""
    tl = text.lower()
    if re.search(r"\bmonotypic\b|\bsole species\b", tl): return "Sole species in its genus."
    if "national fruit" in tl or "national flower" in tl: return "Holds national status in some countries."
    if "garland" in tl or "leis" in tl: return "Often used in ceremonial garlands."
    if "cultivar" in tl or "landrace" in tl: return "Known for many cultivars and local varieties."
    if "butterfly" in tl or "larvae" in tl or "host plant" in tl: return "Supports certain butterfly species."
    m = re.search(r"\bonly [^.]{3,60}\b", text, flags=re.I)
    if m: return m.group(0).strip().rstrip(".") + "."
    return ""

# -----------------------------------------------------------------------------
# Sentence helpers for “Fun facts” & “Story” sections (non-LLM)
# -----------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    if not text: return []
    # simple splitter; keeps things lightweight
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in sents if s.strip()]

_FUN_TRIGGERS = [
    r"\bnational (?:flower|fruit|tree)\b", r"\bmonotypic\b", r"\bonly\b",
    r"\bworld(?:'s)? (?:largest|smallest|tallest)\b", r"\boldest\b", r"\brarest\b",
    r"\bused in (?:ceremon(?:y|ies)|garlands|leis|festivals)\b", r"\bcultivar(?:s)?\b",
    r"\bsacred\b|\bsymbol\b|\bemblem\b", r"\bmedicin(?:e|al)\b", r"\bperfume\b|\bfragrance\b",
]
_STORY_TRIGGERS = [
    r"\blegend\b", r"\bmyth\b", r"\bfolklore\b", r"\bstory\b", r"\bnarrative\b",
    r"\bhistory\b", r"\bethnobotany\b", r"\border of\b", r"\bwas introduced\b", r"\bnamed after\b",
]

def _pick_fun_facts(text: str, max_items: int = 5) -> list[str]:
    """Pick up to N short ‘fun fact’ sentences from the extract."""
    sents = _split_sentences(text)
    out = []
    for s in sents:
        if any(re.search(p, s, flags=re.I) for p in _FUN_TRIGGERS):
            out.append(s.rstrip(".") + ".")
        if len(out) >= max_items:
            break
    return out

def _pick_story_snippet(text: str, max_sents: int = 3) -> str:
    """Pick a short ‘story’ (legend/folklore/history) snippet if present."""
    sents = _split_sentences(text)
    chosen = []
    for s in sents:
        if any(re.search(p, s, flags=re.I) for p in _STORY_TRIGGERS):
            chosen.append(s.rstrip(".") + ".")
            if len(chosen) >= max_sents:
                break
    return " ".join(chosen)

# -----------------------------------------------------------------------------
# Overview builder (non-LLM) — 10–30 sentences, lively, non-repetitive
# -----------------------------------------------------------------------------
def _build_overview_10_30(summary: str, full_text: str, where: str, uses: str, cautions: str, needs: str, funfact_one: str) -> str:
    """
    Build a 10–30 sentence overview by rephrasing summary + sampling distinct,
    non-bullet sentences from the full extract. Avoid long country lists.
    """
    # seed with rephrased opener
    sents = []
    base = [x.strip() for x in summary.strip().split(".") if x.strip()]
    if base:
        opener = re.sub(r"is a (species|genus) of [^.]+", "is a distinctive plant", base[0], flags=re.I)
        sents.append(opener.rstrip(".") + ".")
    else:
        sents.append("A distinctive plant that draws attention when in season.")

    # add short hooks
    if re.search(r"\b(showy|fragrant|large|panicles?|racemes?)\b", summary.lower()):
        sents.append("Up close, its display can be surprisingly showy, from flower form to subtle fragrance.")
    if re.search(r"\blatex\b|\bmilky sap\b|\bexudate\b", (summary + " " + full_text).lower()):
        sents.append("It exudes a milky latex—biologically intriguing, yet worth handling with care.")
    if where:
        sents.append("Within its native range it threads through varied habitats and climates.")
    if uses:
        sents.append("Beyond looks, people have found practical uses for it in everyday or traditional contexts.")
    if needs:
        sents.append("In cultivation, it thrives when basic needs—light, soil and moisture—are matched to its preferences.")
    if funfact_one:
        sents.append(f"A curious tidbit often noted: {funfact_one.rstrip('.')}.")
    # bring a couple extra sentences from summary if useful
    for extra in base[1:3]:
        sents.append(re.sub(r"\s+", " ", extra).rstrip(".") + ".")

    # draw more diverse sentences from full text; filter out long lists and duplicates
    pool = _split_sentences(full_text)
    def _ok(sent: str) -> bool:
        if len(sent) < 30 or len(sent) > 260:  # keep readable length
            return False
        if re.search(r",\s*[A-Z][a-z]+(?:,|\sand\s)", sent):  # reduce country lists
            if len(sent) > 180:
                return False
        key = re.sub(r"[^a-z0-9]+", " ", sent.lower()).strip()
        return key != ""

    bullets_text = " ".join(filter(None, [where, uses, cautions, needs, funfact_one])).lower()
    seen_keys = set(re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() for s in sents)

    for sent in pool:
        if not _ok(sent): 
            continue
        key = re.sub(r"[^a-z0-9]+", " ", sent.lower()).strip()
        if key in seen_keys:
            continue
        # avoid repeating bullet phrases crudely
        if any(tok in bullets_text for tok in key.split()[:6]):
            continue
        sents.append(sent.rstrip(".") + ".")
        seen_keys.add(key)
        if len(sents) >= 30:
            break

    # ensure lower bound
    while len(sents) < 10 and pool:
        s = pool.pop(0)
        if _ok(s):
            sents.append(s.rstrip(".") + ".")

    return " ".join(sents[:30])

# -----------------------------------------------------------------------------
# Card builders (non-LLM and LLM)
# -----------------------------------------------------------------------------
def _microcard_no_llm(label: str, confidence: float, facts: dict, full_text: str) -> str:
    """
    Build full card (Quick facts + Fun facts + Story + Overview 10–30 + Sources)
    without relying on the LLM. Uses heuristics + the full extract.
    """
    title = facts.get("title") or label.replace("_", " ")
    summary = facts.get("summary") or ""
    url = facts.get("url") or f"https://en.wikipedia.org/wiki/{quote(label.replace(' ', '_'))}"
    corpus = (full_text or "") + "\n" + summary

    # facts
    commons   = _heuristic_common_names(corpus)
    where     = _heuristic_where_found(corpus)
    flowering = _find_months_near(corpus, "flower")
    fruiting  = _find_months_near(corpus, "fruit") or _season_phrase(corpus, "fruit")
    season    = _heuristic_season(corpus)
    needs     = _heuristic_growing_needs(corpus)
    uses      = _heuristic_uses(corpus)
    cautions  = _heuristic_cautions(corpus)

    # fun facts & story
    funfact_one  = _heuristic_fun_fact(corpus)
    funfacts_more = _pick_fun_facts(corpus, max_items=5)
    story = _pick_story_snippet(corpus, max_sents=3)

    # Quick facts bullets (ordered by relevance/interest)
    bullets = []
    if uses:      bullets.append(f"Uses: {_safe(uses)}")
    if cautions:  bullets.append(f"Cautions: {_safe(cautions)}")
    if commons:   bullets.append(f"Common names: {_safe(', '.join(commons))}")
    if flowering: bullets.append(f"Flowering: {_safe(flowering)}")
    if fruiting:  bullets.append(f"Fruiting: {_safe(fruiting)}")
    if season:    bullets.append(f"Season: {_safe(season)}")
    if needs:     bullets.append(f"Growing needs: {_safe(needs)}")
    if where:     bullets.append(f"Where found: {_safe(where)}")

    overview = _build_overview_10_30(summary, corpus, where, uses, cautions, needs, funfact_one)

    # HTML
    parts = []
    parts.append("<div>")
    parts.append(f"<h3>{_safe(title)}</h3>")

    if bullets:
        parts.append("<h4>Quick facts</h4>")
        parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets[:12]) + "</ul>")

    if funfacts_more:
        parts.append("<h4>Fun facts</h4>")
        parts.append("<ul>" + "".join(f"<li>{_safe(ff)}</li>" for ff in funfacts_more) + "</ul>")

    if story:
        parts.append("<h4>Story</h4>")
        parts.append(f"<p>{_safe(story)}</p>")

    parts.append("<h4>Overview</h4>")
    parts.append(f"<p>{_safe(overview)}</p>")

    if confidence < 0.30:
        parts.append('<p><small>⚠️ Low confidence — double-check the ID.</small></p>')

    parts.append("<h4>Sources</h4>")
    parts.append(f'<ul><li><a href="{_safe(url)}" target="_blank" rel="noreferrer">Wikipedia</a></li></ul>')
    parts.append("</div>")
    return "".join(parts)

def generate_html(label: str, confidence: float, facts_json: dict) -> str:
    """
    LLM path: friendly, casual, engaging; adds:
      - Quick facts (bulleted)
      - Fun facts (bulleted, 2–5 items if present)
      - Story (short anecdote if present)
      - Overview (10–30 sentences), non-repetitive and lively
      - Sources with a Wikipedia link
    Falls back to the non-LLM builder if the API isn’t available.
    """
    wiki = facts_json or {}
    if not wiki.get("url"):
        wiki = _resolve_wikipedia(label)

    title_for_full = wiki.get("title", label).replace(" ", "_")
    full_text = _wiki_full_extract(title_for_full)

    if not client:
        return _microcard_no_llm(label, confidence, wiki, full_text)

    # Build heuristics to nudge the LLM (must only be used if corroborated by text)
    corpus = (full_text or "") + "\n" + (wiki.get("summary") or "")
    flowering = _find_months_near(corpus, "flower")
    fruiting  = _find_months_near(corpus, "fruit") or _season_phrase(corpus, "fruit")
    uses      = _heuristic_uses(corpus)
    cautions  = _heuristic_cautions(corpus)
    commons   = _heuristic_common_names(corpus)
    needs     = _heuristic_growing_needs(corpus)
    where     = _heuristic_where_found(corpus)
    season    = _heuristic_season(corpus)
    funfacts_more = _pick_fun_facts(corpus, max_items=5)
    story = _pick_story_snippet(corpus, max_sents=3)

    prompt = (
        "You are a botanist and science communicator. Build a warm, casual, engaging plant info card in HTML using ONLY the provided Wikipedia content.\n\n"
        f"Species label: {label}\n"
        f"Model confidence: {confidence:.1%}\n"
        f"Resolved page meta: {json.dumps(wiki, ensure_ascii=False)}\n"
        f"Full extract (plain text; may be truncated):\n'''\n{(full_text or '')[:8000]}\n'''\n\n"
        "Heuristic hints (use only if corroborated by the extract):\n"
        f"- Common names: {commons or '—'}\n- Uses: {uses or '—'}\n- Cautions: {cautions or '—'}\n"
        f"- Flowering: {flowering or '—'}\n- Fruiting: {fruiting or '—'}\n- Season: {season or '—'}\n"
        f"- Growing needs: {needs or '—'}\n- Where found: {where or '—'}\n"
        f"- Fun facts candidates: {funfacts_more or '—'}\n- Story snippet: {story or '—'}\n\n"
        "OUTPUT FORMAT (HTML only; no extra text):\n"
        "- Allowed tags: <div>, <h3>, <h4>, <ul>, <li>, <p>, <a>.\n"
        "- Include sections, in order:\n"
        "  1) <h4>Quick facts</h4> with ONE <ul> including (omit if unknown): Uses; Cautions; Common names; Flowering months; Fruiting months (or season if months absent); Season (annual/perennial, etc.); Growing needs; Where found.\n"
        "  2) <h4>Fun facts</h4> with 2–5 bullet points drawn from verifiable tidbits in the extract (no invention).\n"
        "  3) <h4>Story</h4> (2–4 sentences) if there is a legend/folklore/historical anecdote in the text; otherwise omit.\n"
        "  4) <h4>Overview</h4> as a single paragraph of **10–30 sentences**, lively and non-repetitive. Include a sensory hook, morphology/growth habit, one cultural/culinary note (if present), one ecological interaction, a rephrased safety nudge (if relevant), and a curious tidbit. Do NOT restate the Quick facts or long country lists.\n"
        "  5) <h4>Sources</h4> linking the exact Wikipedia URL as anchor text 'Wikipedia'.\n"
        "- Be strictly faithful to the provided content; do not invent facts. If a field is unknown, omit it.\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be precise, friendly, engaging, and never fabricate beyond the provided Wikipedia content."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2200,   # allow 10–30 sentences + sections
            temperature=0.65,
        )
        html = (resp.choices[0].message.content or "").strip()
        # Ensure a Wikipedia link appears
        if "wikipedia" not in html.lower() or "<a " not in html:
            src = _safe(wiki.get("url") or f"https://en.wikipedia.org/wiki/{quote(label.replace(' ', '_'))}")
            tail = f'<h4>Sources</h4><ul><li><a href="{src}" target="_blank" rel="noreferrer">Wikipedia</a></li></ul>'
            html = (html.replace("</div>", tail + "</div>") if "</div>" in html else f"<div>{html}{tail}</div>")
        # Confidence warning if very low
        if confidence < 0.30 and "<small>" not in html:
            html = html.replace("</div>", '<p><small>⚠️ Low confidence — double-check the ID.</small></p></div>')
        return html
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return _microcard_no_llm(label, confidence, wiki, full_text)

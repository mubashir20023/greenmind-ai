from __future__ import annotations
import os, io, re, traceback, json, uuid, csv, datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, jsonify, request, Response
from PIL import Image

from app.identify import identify
from app.explain import generate_cam_overlay
from app.facts import get_wikipedia, generate_html

# Optional: detector status
try:
    from app.detector import DETECTOR
    status = "available" if DETECTOR.is_available() else f"unavailable: {DETECTOR.reason_unavailable()}"
    print(f"[yolo] detector status: {status}")
except Exception as e:
    print("[yolo] detector import failed:", e)
    DETECTOR = None  # ensure symbol exists

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR    = BASE_DIR / "web" / "static"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

# Feedback storage
_FEEDBACK_ROOT_ENV = os.getenv("FEEDBACK_ROOT", str(BASE_DIR / "data" / "feedback"))
FEEDBACK_ROOT = Path(_FEEDBACK_ROOT_ENV)
FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)

FEEDBACK_CSV = FEEDBACK_ROOT / "feedback.csv"
if not FEEDBACK_CSV.exists():
    with FEEDBACK_CSV.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([
            "id","date","verdict","true_label","notes","model",
            "best_label","best_score","alt_labels","image_relpath"
        ])

def sanitize_html(html: str) -> str:
    return re.sub(r"(?is)<script.*?>.*?</script>", "", html or "")

@app.get("/")
def index():
    return render_template("index.html")

# Helpful diagnostics to compare local vs Render env
@app.get("/diag")
def diag():
    try:
        plant_id_key = bool(os.getenv("PLANT_ID_API_KEY"))
        plantnet_key = bool(os.getenv("PLANTNET_API_KEY"))
    except Exception:
        plant_id_key = plantnet_key = False
    return jsonify({
        "yolo_available": bool(DETECTOR and DETECTOR.is_available()),
        "env": {
            "REQUIRE_DETECT": os.getenv("REQUIRE_DETECT"),
            "YOLO_CONF": os.getenv("YOLO_CONF"),
            "YOLO_MIN_AREA_FRAC": os.getenv("YOLO_MIN_AREA_FRAC"),
            "YOLO_MAX_CROPS": os.getenv("YOLO_MAX_CROPS"),
            "CONFIDENCE_THRESHOLD": os.getenv("CONFIDENCE_THRESHOLD"),
            "MIN_IS_PLANT": os.getenv("MIN_IS_PLANT"),
            "MIN_TOP1_GAP": os.getenv("MIN_TOP1_GAP"),
            "TOPK": os.getenv("TOPK"),
        },
        "api_keys": {
            "plant_id": plant_id_key,
            "plantnet": plantnet_key
        }
    })

@app.post("/identify")
def route_identify():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
        topk = int(os.getenv("TOPK", "3"))

        # ------- Gating knobs -------
        REQUIRE_DETECT = os.getenv("REQUIRE_DETECT", "1") == "1"     # hard block if no plant box
        YOLO_MIN_AREA_FRAC = float(os.getenv("YOLO_MIN_AREA_FRAC", "0.04"))
        YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
        MIN_IS_PLANT = float(os.getenv("MIN_IS_PLANT", "0.50"))      # Plant.id gate
        MIN_CONF = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))  # score gate
        MIN_TOP1_GAP = float(os.getenv("MIN_TOP1_GAP", "0.0"))       # ambiguity gate

        # ------- 1) YOLO hard gate (if available) -------
        W, H = img.size
        if DETECTOR and DETECTOR.is_available():
            try:
                raw = DETECTOR.detect(img)  # [{'cls','conf','box','crop'}]
            except Exception:
                raw = []
            valid = []
            for d in raw:
                (x1, y1, x2, y2) = d.get("box", (0,0,0,0))
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                frac = area / float(W * H) if W*H else 0.0
                if float(d.get("conf", 0.0)) >= YOLO_CONF and frac >= YOLO_MIN_AREA_FRAC:
                    valid.append(d)
            if REQUIRE_DETECT and not valid:
                return jsonify({
                    "no_plant": True,
                    "message": "No plant-like object detected. Try moving closer to leaves/flowers or improve lighting."
                }), 422

        # ------- 2) Run identification -------
        best, alts = identify(img, gallery_root=None, topk=topk)

        # ------- 3) Plant.id is_plant_probability gate (if present) -------
        is_plant_prob = best.get("is_plant_prob")
        if isinstance(is_plant_prob, (int, float)) and is_plant_prob < MIN_IS_PLANT:
            return jsonify({
                "no_plant": True,
                "message": "This photo likely does not contain a plant (API signal). Try another angle/subject."
            }), 422

        # ------- 4) Ambiguity gate (top-1 vs top-2 gap) -------
        if MIN_TOP1_GAP > 0.0 and len(alts) >= 2:
            gap = float(alts[0].get("score", 0.0)) - float(alts[1].get("score", 0.0))
            if gap < MIN_TOP1_GAP:
                return jsonify({
                    "low_confidence": True,
                    "best": best,
                    "alternatives": alts,
                    "model": os.getenv("MODEL_NAME", "unknown"),
                    "message": "Ambiguous result (top-1 too close to top-2). Try another photo."
                }), 200

        # ------- 5) Low-confidence gate -------
        if float(best.get("score", 0.0)) < MIN_CONF:
            return jsonify({
                "low_confidence": True,
                "best": best,
                "alternatives": alts,
                "model": os.getenv("MODEL_NAME", "unknown"),
                "message": "Low-confidence ID. Try a sharper photo with the plant centered."
            }), 200

        # ------- 6) Normal path: build facts (with safe fallback) -------
        facts_html = ""
        try:
            facts = get_wikipedia(best["label"])
            facts_html = sanitize_html(generate_html(best["label"], best["score"], facts))
        except Exception as e:
            # Never let facts generation break the flow
            facts_html = f"<p>Facts temporarily unavailable. ({str(e)})</p>"

        model_name = os.getenv("MODEL_NAME", "resnet18")
        return jsonify({
            "best": best,
            "alternatives": alts,
            "facts_html": facts_html,
            "model": model_name
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Identify failed: {e}"}), 500

@app.post("/explain")
def explain():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
        method = request.args.get("method", "gradcam")  # 'gradcam' or 'eigen'
        png_bytes = generate_cam_overlay(img, method=method)
        return Response(png_bytes, mimetype="image/png")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Explain failed: {e}"}), 500

@app.post("/feedback")
def feedback():
    try:
        meta_raw = request.form.get("meta")
        if not meta_raw:
            return jsonify({"error": "Missing meta"}), 400
        try:
            meta = json.loads(meta_raw)
        except Exception:
            return jsonify({"error": "Invalid meta JSON"}), 400

        img_file = request.files.get("file")
        if img_file is None:
            return jsonify({"error": "Missing image file"}), 400

        date_str = datetime.date.today().isoformat()
        fid = str(uuid.uuid4())
        out_dir = FEEDBACK_ROOT / date_str / fid
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = out_dir / "image.jpg"
        img_file.save(img_path)

        meta_path = out_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        best_label = None
        best_score = None
        if isinstance(meta.get("best"), dict):
            best_label = meta["best"].get("label")
            best_score = meta["best"].get("score")

        alt_labels = []
        for a in (meta.get("alternatives") or []):
            if isinstance(a, dict) and "label" in a:
                alt_labels.append(a["label"])

        with FEEDBACK_CSV.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                fid,
                date_str,
                meta.get("verdict"),
                meta.get("true_label") or "",
                meta.get("notes") or "",
                meta.get("model") or "",
                best_label or "",
                f"{best_score:.6f}" if isinstance(best_score, (float,int)) else "",
                ";".join(alt_labels),
                str(img_path.relative_to(FEEDBACK_ROOT))
            ])

        return jsonify({"status": "ok", "id": fid})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

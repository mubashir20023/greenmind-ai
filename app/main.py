# app/main.py
from __future__ import annotations
import os, io, re, traceback, json, uuid, csv, datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from PIL import Image

from app.identify import identify
from app.explain import generate_cam_overlay
from app.facts import get_wikipedia, generate_html

# Optional: log detector status if present
try:
    from app.detector import DETECTOR
    status = "available" if DETECTOR.is_available() else f"unavailable: {DETECTOR.reason_unavailable()}"
    print(f"[yolo] detector status: {status}")
except Exception as e:
    print("[yolo] detector import failed:", e)

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR    = BASE_DIR / "web" / "static"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

# Feedback storage
FEEDBACK_ROOT = BASE_DIR / "data" / "feedback"
FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)
FEEDBACK_CSV = FEEDBACK_ROOT / "feedback.csv"
if not FEEDBACK_CSV.exists():
    with FEEDBACK_CSV.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([
            "id","date","verdict","true_label","notes","model",
            "best_label","best_score","alt_labels","image_relpath"
        ])

def sanitize_html(html: str) -> str:
    return re.sub(r"(?is)<script.*?>.*?</script>", "", html)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/identify")
def route_identify():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
        topk = int(os.getenv("TOPK", "3"))

        best, alts = identify(img, gallery_root=None, topk=topk)
        facts = get_wikipedia(best["label"])
        html = sanitize_html(generate_html(best["label"], best["score"], facts))
        model_name = os.getenv("MODEL_NAME", "resnet18")

        return jsonify({
            "best": best,
            "alternatives": alts,
            "facts_html": html,
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

from __future__ import annotations
from app.database import SessionLocal  # ✅ Only import SessionLocal from database
from app.models import User, Plant, PlantImage, PlantAdvice  # ✅ Import User from models
import os, io, re, traceback, json, uuid, csv, datetime, cv2, base64
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
load_dotenv()
from flask_login import LoginManager, login_required, login_user, logout_user, current_user

from flask import Flask, render_template, jsonify, request, Response, send_from_directory
from PIL import Image

# App modules
from app.identify import identify
from app.explain import generate_cam_overlay
from app.facts import get_wikipedia, generate_html
from app.healthcare import render_care_html
from app.health import assess_health as local_health

# Optional: detector status (YOLO hard gate)
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
STATIC_DIR = BASE_DIR / "web" / "static"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())
# ---------------- Login Manager ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        return db.query(User).get(int(user_id))
    finally:
        db.close()

# Feedback storage
_FEEDBACK_ROOT_ENV = os.getenv("FEEDBACK_ROOT", str(BASE_DIR / "data" / "feedback"))
FEEDBACK_ROOT = Path(_FEEDBACK_ROOT_ENV)
FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)

FEEDBACK_CSV = FEEDBACK_ROOT / "feedback.csv"
if not FEEDBACK_CSV.exists():
    with FEEDBACK_CSV.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([
            "id", "date", "verdict", "true_label", "notes", "model",
            "best_label", "best_score", "alt_labels", "image_relpath"
        ])


def sanitize_html(html_text: str) -> str:
    return re.sub(r"(?is)<script.*?>.*?</script>", "", html_text or "")


# ---------------------- Pages ----------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(email=email).first()
            if user:
                flash("Email already exists!", "error")
                return redirect(url_for("signup"))
            
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(name=name, email=email, password_hash=hashed_password)
            db.add(new_user)
            db.commit()
            flash("Account created successfully!", "success")
            return redirect(url_for("login"))
        finally:
            db.close()
    
    return render_template("signup.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)  # ✅ Actually log them in!
                flash("Login successful!", "success")
                return redirect(url_for("index"))
            else:
                flash("Invalid credentials", "error")
                return redirect(url_for("login"))
        finally:
            db.close()
    
    return render_template("login.html")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health-secondary")
def health_secondary_page():
    # dedicated Health page
    return render_template("health.html")


# Serve CAM images under runs/health/*
@app.get("/runs/health/<path:fname>")
def serve_health_runs(fname: str):
    return send_from_directory(str(BASE_DIR / "runs" / "health"), fname)


# ---------------------- Diagnostics ----------------------


@app.get("/diag")
def diag():
    try:
        plant_id_key = bool(os.getenv("PLANT_ID_API_KEY") or os.getenv("PLANTID_API_KEY"))
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
            # health-related knobs
            "HEALTH_MODE": os.getenv("HEALTH_MODE"),
            "HEALTH_CONF": os.getenv("HEALTH_CONF"),
            "USE_PLANTID_HEALTH": os.getenv("USE_PLANTID_HEALTH"),
        },
        "api_keys": {
            "plant_id": plant_id_key,
            "plantnet": plantnet_key
        }
    })


# ---------------------- Identify ----------------------


@app.post("/identify")
def route_identify():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
        topk = int(os.getenv("TOPK", "3"))

        # ------- Gating knobs -------
        REQUIRE_DETECT = os.getenv("REQUIRE_DETECT", "1") == "1"
        YOLO_MIN_AREA_FRAC = float(os.getenv("YOLO_MIN_AREA_FRAC", "0.04"))
        YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
        MIN_IS_PLANT = float(os.getenv("MIN_IS_PLANT", "0.60"))
        MIN_CONF = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
        MIN_TOP1_GAP = float(os.getenv("MIN_TOP1_GAP", "0.0"))

        # ------- 1) YOLO hard gate (if available) -------
        W, H = img.size
        if DETECTOR and DETECTOR.is_available():
            try:
                raw = DETECTOR.detect(img)  # [{'cls','conf','box','crop'}]
            except Exception:
                raw = []
            valid = []
            for d in raw:
                (x1, y1, x2, y2) = d.get("box", (0, 0, 0, 0))
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                frac = area / float(W * H) if W * H else 0.0
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
                "message": "This photo does not appear to contain a plant.",
                "model": os.getenv("MODEL_NAME", "unknown"),
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

        # ------- 6) Normal path: build facts -------
        facts_html = ""
        try:
            facts = get_wikipedia(best["label"])
            facts_html = sanitize_html(generate_html(best["label"], best["score"], facts))
        except Exception as e:
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


# ---------------------- Explain (Grad/Eigen-CAM) ----------------------


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


# ---------------------- Feedback ----------------------


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
                f"{best_score:.6f}" if isinstance(best_score, (float, int)) else "",
                ";".join(alt_labels),
                str(img_path.relative_to(FEEDBACK_ROOT))
            ])

        return jsonify({"status": "ok", "id": fid})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------- Health API ----------------------


@app.post("/api/health")
def api_health():
    """
    Health flow:
      1) Read image (file or base64).
      2) YOLO hard gate (same as /identify).
      3) Get plant name via identify() for gating + UI.
      4) Run local health model via app.health.assess_health(bgr).
      5) Return unified JSON for the Health page and /api/health/care.
    """
    # -------- 0) Read image (multipart or JSON base64) --------
    raw = None
    if "image" in request.files:
        raw = request.files["image"].read()
    else:
        data = request.get_json(force=True, silent=True) or {}
        b64 = str(data.get("image") or "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
        except Exception:
            raw = None

    if not raw:
        return jsonify({"error": "No image"}), 400

    # Pillow + OpenCV views
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    # -------- 1) YOLO hard gate (same knobs as /identify) --------
    REQUIRE_DETECT = os.getenv("REQUIRE_DETECT", "1") == "1"
    YOLO_MIN_AREA_FRAC = float(os.getenv("YOLO_MIN_AREA_FRAC", "0.04"))
    YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
    W, H = pil.size

    if DETECTOR and DETECTOR.is_available():
        try:
            raw_dets = DETECTOR.detect(pil)  # [{'cls','conf','box','crop'}]
        except Exception:
            raw_dets = []
        valid = []
        for d in raw_dets:
            (x1, y1, x2, y2) = d.get("box", (0, 0, 0, 0))
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            frac = area / float(W * H) if W * H else 0.0
            if float(d.get("conf", 0.0)) >= YOLO_CONF and frac >= YOLO_MIN_AREA_FRAC:
                valid.append(d)
        if REQUIRE_DETECT and not valid:
            return jsonify({
                "no_plant": True,
                "message": "No plant-like object detected. Try moving closer to leaves/flowers or improve lighting."
            }), 422

    # -------- 2) Plant name via identify() ensemble (for gating + UI) --------
    MIN_IS_PLANT = float(os.getenv("MIN_IS_PLANT", "0.60"))
    TOPK = int(os.getenv("TOPK", "3"))

    plant_name = None
    is_plant_prob = None
    plant_conf = None

    try:
        best, alts = identify(pil, gallery_root=None, topk=TOPK)
        plant_name = best.get("label") or "Unknown"
        is_plant_prob = best.get("is_plant_prob")
        plant_conf = float(best.get("score", 0.0))

        # same gate as /identify
        if isinstance(is_plant_prob, (int, float)) and is_plant_prob < MIN_IS_PLANT:
            return jsonify({
                "no_plant": True,
                "message": "This photo does not appear to contain a plant.",
                "model": os.getenv("MODEL_NAME", "unknown"),
            }), 422
    except Exception:
        # plant name is best-effort; health model can still run
        pass

    # -------- 3) Local health model (YOLO + classifier + identify) --------
    try:
        health_result = local_health(bgr)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Health model failed: {e}"}), 500

    # Merge identification info (prefer the identify() result for UI)
    if plant_name:
        health_result["plant_name"] = plant_name
    if plant_conf is not None:
        health_result["plant_confidence"] = plant_conf

    # Extract fields with safe defaults
    status = (health_result.get("status") or "unknown").lower()
    confidence = float(health_result.get("confidence") or 0.0)
    diseases = health_result.get("diseases") or []
    crops = health_result.get("crops") or []
    external = health_result.get("external") or {}

    final_plant_name = health_result.get("plant_name") or plant_name or "Unknown"

    return jsonify({
        # for existing UI shape
        "plant": {
            "name": final_plant_name,
            "is_plant_probability": is_plant_prob,
        },
        # flat fields (used by /api/health/care)
        "plant_name": final_plant_name,
        "plant_confidence": health_result.get("plant_confidence") or plant_conf,
        "status": status,
        "confidence": confidence,
        "diseases": diseases,
        "crops": crops,
        "external": external,
    })


# ---------------------- Health Care (tips via LLM) ----------------------


@app.post("/api/health/care")
def api_health_care():
    data = request.get_json(force=True, silent=True) or {}

    status = (data.get("status") or "unknown").lower()
    diseases = data.get("diseases") or []

    # Try to read plant info in a forgiving way:
    plant_name = None
    plant_confidence = None
    overall_confidence = None

    # 1) flat fields: { plant_name, plant_confidence, confidence }
    if data.get("plant_name"):
        plant_name = str(data["plant_name"])
    if isinstance(data.get("plant_confidence"), (int, float)):
        plant_confidence = float(data["plant_confidence"])
    if isinstance(data.get("confidence"), (int, float)):
        overall_confidence = float(data["confidence"])

    # 2) nested "plant": { name, is_plant_probability }
    plant_block = data.get("plant") or {}
    if not plant_name and plant_block.get("name"):
        plant_name = str(plant_block["name"])
    if plant_confidence is None and isinstance(plant_block.get("is_plant_probability"), (int, float)):
        plant_confidence = float(plant_block["is_plant_probability"])

    # Normalize diseases list: accept [{"name":..,"prob":..}] or ["powdery_mildew", ...]
    norm = []
    for d in diseases:
        if isinstance(d, dict) and "name" in d:
            try:
                prob = float(d.get("prob", d.get("score", 0)))
            except Exception:
                prob = 0.0
            norm.append({"name": d["name"], "prob": prob})
        else:
            norm.append({"name": str(d), "prob": 0.0})

    html_out = render_care_html(
        status=status,
        diseases=norm,
        plant_name=plant_name,
        plant_confidence=plant_confidence,
        overall_confidence=overall_confidence,
    )
    return jsonify({"html": html_out})


# ---------------------- Entrypoint ----------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

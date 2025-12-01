# app/main.py
from __future__ import annotations

import os
import io
import re
import time
import json
import uuid
import csv
import traceback
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
# from app.models import db

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    g,
    jsonify,
    Response,
    send_from_directory,
)
from werkzeug.security import generate_password_hash, check_password_hash

from flask_login import (
    LoginManager,
    login_required,
    login_user,
    logout_user,
    current_user,
)

# SQL / models
from app.database import SessionLocal
from app.models import User, Plant, PlantPhoto, PlantAdvice, PlantHealth

# App-specific helpers
from app.identify import identify
from app.explain import generate_cam_overlay
from app.facts import get_wikipedia, generate_html
from app.healthcare import render_care_html
from app.health import assess_health as local_health

# Backend helpers (pure functions/classes — no Flask decorators)
from app.backends import try_backends, PlantIdBackend, PlantNetBackend, health_assess
from app.utils.common_group import normalize_group
load_dotenv()

# Optional: detector status (YOLO gate)
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
        csv.writer(f).writerow(
            [
                "id",
                "date",
                "verdict",
                "true_label",
                "notes",
                "model",
                "best_label",
                "best_score",
                "alt_labels",
                "image_relpath",
            ]
        )


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

            hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
            new_user = User(name=name, email=email, password_hash=hashed_password)
            db.add(new_user)
            db.commit()
            flash("Account created successfully!", "success")
            return redirect(url_for("login"))
        finally:
            db.close()

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        db = SessionLocal()
        try:
            user = db.query(User).filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
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
    return jsonify(
        {
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
                "HEALTH_MODE": os.getenv("HEALTH_MODE"),
                "HEALTH_CONF": os.getenv("HEALTH_CONF"),
                "USE_PLANTID_HEALTH": os.getenv("USE_PLANTID_HEALTH"),
            },
            "api_keys": {"plant_id": plant_id_key, "plantnet": plantnet_key},
        }
    )

# ===========================================================
#  GET LATEST HEALTH ENTRY
# ===========================================================
@app.route("/api/get_latest_health")
@login_required
def api_get_latest_health():
    plant_id = request.args.get("plant_id")
    if not plant_id:
        return jsonify({"error": "missing plant_id"}), 400

    try:
        session = SessionLocal()
        # Fetch last health entry
        last = (
            session.query(PlantHealth)
            .filter(PlantHealth.plant_id == plant_id)
            .order_by(PlantHealth.created_at.desc())
            .first()
        )

        if not last:
            return jsonify({"error": "no health data"}), 404

        try:
            payload = json.loads(last.payload)
        except:
            payload = {}

        status = payload.get("status") or payload.get("health_status") or "unknown"
        confidence = payload.get("confidence") or payload.get("confidence_percent") or 0
        raw_diseases = payload.get("diseases", [])
        plant = session.query(Plant).filter(Plant.id == plant_id).first()
        if not plant:
            return jsonify({"error": "invalid plant_id"}), 404
        plant_name = (plant.name or "").lower()
        plant_species = (plant.species or "").lower()

        # Filter diseases that contain plant-specific keywords
        filtered = []
        for d in raw_diseases:
            name = ""
            if isinstance(d, str):
                name = d.lower()
            elif isinstance(d, dict):
                name = (d.get("name") or "").lower()

            # RULE 1 — skip tomato diseases for non-tomato plants
            if "tomato" in name and "tomato" not in plant_name and "tomato" not in plant_species:
                continue

            # RULE 2 — skip potato-only diseases
            if "blight" in name and "potato" in name and "potato" not in plant_name:
                continue

            filtered.append(d)

        diseases = filtered


        # Care tips (optional)
        care_html = payload.get("care_html", "")

        return jsonify({
            "status": status,
            "score": int(float(confidence) * 100) if isinstance(confidence, float) else int(confidence),
            "diseases": diseases,
            "care_html": care_html
        })

    except Exception as e:
        print("health fetch error:", e)
        return jsonify({"error": "server error"}), 500

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
                return (
                    jsonify(
                        {
                            "no_plant": True,
                            "message": "No plant-like object detected. Try moving closer to leaves/flowers or improve lighting.",
                        }
                    ),
                    422,
                )

        # ------- 2) Run identification -------
        best, alts = identify(img, gallery_root=None, topk=topk)

        # ------- 3) Plant.id is_plant_probability gate (if present) -------
        is_plant_prob = best.get("is_plant_prob")
        if isinstance(is_plant_prob, (int, float)) and is_plant_prob < MIN_IS_PLANT:
            return (
                jsonify(
                    {
                        "no_plant": True,
                        "message": "This photo does not appear to contain a plant.",
                        "model": os.getenv("MODEL_NAME", "unknown"),
                    }
                ),
                422,
            )

        # ------- 4) Ambiguity gate (top-1 vs top-2 gap) -------
        if MIN_TOP1_GAP > 0.0 and len(alts) >= 2:
            gap = float(alts[0].get("score", 0.0)) - float(alts[1].get("score", 0.0))
            if gap < MIN_TOP1_GAP:
                return (
                    jsonify(
                        {
                            "low_confidence": True,
                            "best": best,
                            "alternatives": alts,
                            "model": os.getenv("MODEL_NAME", "unknown"),
                            "message": "Ambiguous result (top-1 too close to top-2). Try another photo.",
                        }
                    ),
                    200,
                )

        # ------- 5) Low-confidence gate -------
        if float(best.get("score", 0.0)) < MIN_CONF:
            return (
                jsonify(
                    {
                        "low_confidence": True,
                        "best": best,
                        "alternatives": alts,
                        "model": os.getenv("MODEL_NAME", "unknown"),
                        "message": "Low-confidence ID. Try a sharper photo with the plant centered.",
                    }
                ),
                200,
            )

        # ------- 6) Normal path: build facts -------
        facts_html = ""
        try:
            facts = get_wikipedia(best["label"])
            facts_html = sanitize_html(generate_html(best["label"], best["score"], facts))
        except Exception as e:
            facts_html = f"<p>Facts temporarily unavailable. ({str(e)})</p>"

        model_name = os.getenv("MODEL_NAME", "resnet18")
        return jsonify({"best": best, "alternatives": alts, "facts_html": facts_html, "model": model_name})
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

        date_str = datetime.today().date().isoformat()
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
            csv.writer(f).writerow(
                [
                    fid,
                    date_str,
                    meta.get("verdict"),
                    meta.get("true_label") or "",
                    meta.get("notes") or "",
                    meta.get("model") or "",
                    best_label or "",
                    f"{best_score:.6f}" if isinstance(best_score, (float, int)) else "",
                    ";".join(alt_labels),
                    str(img_path.relative_to(FEEDBACK_ROOT)),
                ]
            )

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
            return (
                jsonify(
                    {
                        "no_plant": True,
                        "message": "No plant-like object detected. Try moving closer to leaves/flowers or improve lighting.",
                    }
                ),
                422,
            )

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
            return (
                jsonify(
                    {
                        "no_plant": True,
                        "message": "This photo does not appear to contain a plant.",
                        "model": os.getenv("MODEL_NAME", "unknown"),
                    }
                ),
                422,
            )
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

    return jsonify(
        {
            "plant": {"name": final_plant_name, "is_plant_probability": is_plant_prob},
            "plant_name": final_plant_name,
            "plant_confidence": health_result.get("plant_confidence") or plant_conf,
            "status": status,
            "confidence": confidence,
            "diseases": diseases,
            "crops": crops,
            "external": external,
        }
    )


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
        status=status, diseases=norm, plant_name=plant_name, plant_confidence=plant_confidence, overall_confidence=overall_confidence
    )
    return jsonify({"html": html_out})


# ---------------------- Dashboard ----------------------
@app.get("/my-plants")
@login_required
def my_plants_page():
    return render_template("my_plants.html")


@app.get("/api/myplants")
@login_required
def api_myplants():
    db = SessionLocal()
    try:
        plants = db.query(Plant).filter_by(user_id=current_user.id).all()
        out = []

        for p in plants:
            # Last image (existing code)
            last_img = db.query(PlantPhoto) \
                .filter_by(plant_id=p.id) \
                .order_by(PlantPhoto.uploaded_on.desc()) \
                .first()

            # Last health (existing code)
            last_health = db.query(PlantHealth) \
                .filter_by(plant_id=p.id) \
                .order_by(PlantHealth.created_at.desc()) \
                .first()
            last_health_j = json.loads(last_health.payload) if last_health else None

            # History (return last N health records)
            health_rows = db.query(PlantHealth) \
                .filter_by(plant_id=p.id) \
                .order_by(PlantHealth.created_at.desc()) \
                .limit(10) \
                .all()

            history = []
            for hr in health_rows:
                try:
                    payload = json.loads(hr.payload)
                except Exception:
                    payload = {"raw": hr.payload}
                history.append({
                    "id": hr.id,
                    "time": hr.created_at.isoformat() if hr.created_at else None,
                    "payload": payload
                })

            # Recent photos (optional; you already have /api/plant/photos as separate endpoint)
            photo_rows = db.query(PlantPhoto) \
                .filter_by(plant_id=p.id) \
                .order_by(PlantPhoto.uploaded_on.desc()) \
                .limit(6) \
                .all()
            photos = [{"url": r.photo_url, "time": r.uploaded_on.isoformat() if r.uploaded_on else None} for r in photo_rows]

            out.append({
                "id": p.id,
                "name": p.name,
                "species": p.species,
                "group": p.common_group,
                "description": p.description or "",
                "last_image": (last_img.photo_url if last_img else None),
                "last_health": last_health_j,
                "last_health_time": last_health.created_at.isoformat() if last_health else None,
                "history": history,
                "photos": photos
            })
        return jsonify({"plants": out, "count": len(out)})
    finally:
        db.close()


# Serve user uploaded plant images
@app.route("/data/user_plants/<path:filename>")
@login_required 
def serve_plant_photo(filename):
    base = BASE_DIR / "data" / "user_plants"
    return send_from_directory(base, filename)


def normalize_group_name(label: str):
    """
    Converts plant label to a stable group name.
    Examples:
      'Mango Tree' -> 'Mango'
      'Cavendish Banana Plant' -> 'Banana'
      'Rose (Hybrid Tea)' -> 'Rose'
    """
    label = label.lower()
    words = label.split()
    # Keep only first meaningful word
    for w in words:
        if w.isalpha():
            return w.title()
    return label.title()

# ---------------------- API: save a scan result (identify/health) ----------------------
@app.post("/api/save-scan")
@login_required
def api_save_scan():
    """
    Accepts multipart/form-data:
      - plant_id (optional)
      - plant_name (required when creating new plant)
      - species (optional)
      - image file
      - health_json (AI health result)
    """
    db = SessionLocal()
    try:
        plant_id = request.form.get("plant_id")
        plant_name = (request.form.get("plant_name") or "").strip()
        species_name = (request.form.get("species") or plant_name).strip()
        health_raw = request.form.get("health_json")
        img_file = request.files.get("image")

        # ---------------------------
        # Normalize group name helper
        # ---------------------------
        def normalize_group(label: str):
            import re
            if not label:
                return "Unknown"
            label = label.lower().strip()
            parts = re.split(r"[ _\-]+", label)
            first = parts[0]

            SPECIAL = {
                "musa": "Banana",
                "mangifera": "Mango",
                "rosa": "Rose",
                "hibiscus": "Hibiscus",
                "aloe": "Aloe",
                "ocimum": "Basil",
            }
            if first in SPECIAL:
                return SPECIAL[first]
            return first.title()

        # =====================================================
        # 1) CREATE NEW PLANT (NO MERGING — ALWAYS NEW ENTRY)
        # =====================================================
        if not plant_id:
            if not plant_name:
                return jsonify({"error": "Missing plant_name"}), 400

            plant = Plant(
                name=plant_name,
                species=species_name,
                common_group=normalize_group(species_name),
                description="",
                user_id=current_user.id
            )
            db.add(plant)
            db.commit()
            db.refresh(plant)
            plant_id = plant.id

        # =====================================================
        # 2) EXISTING PLANT → ONLY update species/group if empty
        # =====================================================
        else:
            plant = (
                db.query(Plant)
                .filter_by(id=int(plant_id), user_id=current_user.id)
                .first()
            )
            if not plant:
                return jsonify({"error": "Plant not found"}), 404

            # DO NOT override group/species unless missing
            if species_name:
                if not plant.species:
                    plant.species = species_name
                if not plant.common_group:
                    plant.common_group = normalize_group(species_name)

        # =====================================================
        # 3) SAVE HEALTH PAYLOAD
        # =====================================================
        payload = None
        if health_raw:
            try:
                payload = json.loads(health_raw)
            except Exception:
                payload = {"raw": str(health_raw)}

            ph = PlantHealth(
                plant_id=plant_id,
                payload=json.dumps(payload)
            )
            db.add(ph)

        # =====================================================
        # 4) SAVE IMAGE
        # =====================================================
        img_rel = None
        if img_file:
            base = Path(BASE_DIR) / "data" / "user_plants" / str(current_user.id) / str(plant_id)
            base.mkdir(parents=True, exist_ok=True)

            filename = f"{int(datetime.utcnow().timestamp())}.jpg"
            outp = base / filename
            img_file.save(str(outp))

            img_rel = str(outp.relative_to(BASE_DIR)).replace("\\", "/")

            pi = PlantPhoto(
                plant_id=plant_id,
                photo_url=img_rel,
                uploaded_on=datetime.utcnow(),
                health_status=payload.get("status") if isinstance(payload, dict) else None
            )
            db.add(pi)

        db.commit()

        return jsonify({
            "status": "ok",
            "plant_id": int(plant_id),
            "image_path": img_rel
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        db.close()

# ---------------------- API: plant image upload (base64) ----------------------
@app.post("/api/plant/image")
@login_required
def api_save_plant_image():
    """
    Save plant image uploaded as base64.
    Body JSON:
        { "plant_id": 123, "image": "data:image/jpeg;base64,...." }
    """
    db = SessionLocal()
    try:
        data = request.get_json(force=True, silent=True) or {}
        plant_id = data.get("plant_id")
        img64 = data.get("image")

        if not plant_id or not img64:
            return jsonify({"error": "Missing plant_id or image"}), 400

        # Decode base64 image
        try:
            # accept with or without the data: prefix
            if "," in img64:
                img_b64 = img64.split(",", 1)[1]
            else:
                img_b64 = img64
            img_data = base64.b64decode(img_b64)
        except Exception:
            return jsonify({"error": "Invalid base64 image"}), 400

        # Create folder: /data/user_plants/<user_id>/<plant_id>/
        base = Path(BASE_DIR) / "data" / "user_plants" / str(current_user.id) / str(plant_id)
        base.mkdir(parents=True, exist_ok=True)

        filename = f"{int(time.time())}.jpg"
        filepath = base / filename

        with open(filepath, "wb") as f:
            f.write(img_data)

        rel_path = str(filepath.relative_to(BASE_DIR)).replace("\\", "/")

        # Save to database (timestamp field used if model has it)
        rec = PlantPhoto(
            plant_id=plant_id,
            photo_url=rel_path,
            uploaded_on=datetime.utcnow(),
            health_status=None
        )
        db.add(rec)
        db.commit()

        return jsonify({"ok": True, "image_path": rel_path})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ---------------------- Weather API ----------------------
@app.get("/api/weather")
def api_weather():
    """
    Supports:
        /api/weather?city=Karachi
        /api/weather?lat=...&lon=...
    """
    key = os.getenv("OPENWEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")
    if not key:
        return jsonify({"error": "No API key configured"}), 500

    lat = request.args.get("lat")
    lon = request.args.get("lon")
    city = request.args.get("city")

    try:
        if lat and lon:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
        else:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"q": city or "Karachi", "appid": key, "units": "metric"}

        r = requests.get(url, params=params, timeout=8)
        data = r.json()

        return jsonify({
            "name": data.get("name"),
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "weather": data["weather"],
            "condition": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------- Plant Precautions ----------------------
@app.get("/api/plant_precautions")
@login_required
def api_plant_precautions():
    """
    Query parameters:
      - plant_id (required)
      - weather (optional) - JSON string or omitted
    Returns: { html: "<p>...</p>" }
    """
    plant_id = request.args.get("plant_id")
    if not plant_id:
        return jsonify({"error": "missing plant_id"}), 400

    db = SessionLocal()
    try:
        ph = db.query(PlantHealth).filter_by(plant_id=int(plant_id)).order_by(PlantHealth.created_at.desc()).first()
        plant = db.query(Plant).filter_by(id=int(plant_id), user_id=current_user.id).first()
        if not plant:
            return jsonify({"error": "plant not found"}), 404

        latest = json.loads(ph.payload) if ph else {}
        status = latest.get("status") or "unknown"
        diseases = latest.get("diseases") or []

        weather_str = request.args.get("weather") or None
        weather = None
        if weather_str:
            try:
                weather = json.loads(weather_str)
            except Exception:
                weather = {"note": str(weather_str)}

        try:
            html = render_care_html(status=status, diseases=diseases, plant_name=plant.name, plant_confidence=latest.get("plant_confidence"), overall_confidence=latest.get("confidence"))
        except Exception as e:
            html = f"<p class='muted'>Care tips unavailable: {e}</p>"

        try:
            from app.healthcare import client as openai_client
            if openai_client is not None and weather:
                wtxt = f"Weather: {weather.get('weather',[{}])[0].get('description','')}, temp: {weather.get('main',{}).get('temp')}"
                user = (
                    f"Plant: {plant.name}\n"
                    f"Latest health status: {status}\n"
                    f"Likely diseases: {', '.join([d.get('name','') for d in diseases])}\n"
                    f"{wtxt}\n\n"
                    "Produce one short paragraph (2-4 sentences) with practical precautions specifically for this plant given the weather. Keep it safe and non-prescriptive, focusing on watering, humidity, immediate checks, and a quick next step."
                )
                resp = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "system", "content": "You are a concise plant-care assistant."}, {"role": "user", "content": user}],
                    temperature=0.3,
                )
                extra = (resp.choices[0].message.content or "").strip()
                html = html + f"<div class='card-sub'><h4>Weather-specific note</h4><p>{extra}</p></div>"
        except Exception:
            pass

        return jsonify({"html": html})
    finally:
        db.close()

@app.get("/api/plant/photos")
@login_required
def api_plant_photos():
    plant_id = request.args.get("plant_id")
    if not plant_id:
        return jsonify({"error": "missing plant_id"}), 400

    db = SessionLocal()
    try:
        rows = db.query(PlantPhoto)\
                 .filter_by(plant_id=int(plant_id))\
                 .order_by(PlantPhoto.uploaded_on.desc())\
                 .all()

        return jsonify({
            "photos": [
                {
                    "url": r.photo_url,
                    "time": r.uploaded_on.isoformat() if r.uploaded_on else None
                }
                for r in rows
            ]
        })
    finally:
        db.close()

# ---------------------- Logout + Entrypoint ----------------------
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

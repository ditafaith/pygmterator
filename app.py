import os
import io
import re
import math
import hmac
import sqlite3
import json
import time
import secrets
import tempfile
import base64
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import numpy as np
import pygmt
from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional packages: the server still runs without them; only the related features are disabled
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import contourpy
except ImportError:
    contourpy = None

app = Flask(__name__)
CORS(app)

# Smithsonian GVP volcano database
VOLCANO_FILE = "volcanoes-2026-11-06_20-20-10_+0700.tsv"
FAULT_FILE = "PB2002_boundaries.json"
PUSGEN_FILE = "pusgen_fault_2024.txt"
FAULT_URL = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"

def load_volcano_data():
    if os.path.exists(VOLCANO_FILE):
        try:
            df = pd.read_csv(VOLCANO_FILE, sep="\t")
            df = df.dropna(subset=['Latitude', 'Longitude'])
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            return df.dropna(subset=['Latitude', 'Longitude'])
        except Exception as e:
            print(f"Error loading volcano TSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

df_volcanoes = load_volcano_data()
print(f"Loaded {len(df_volcanoes)} volcanoes from GVP database.")

def load_pusgen_faults():
    """Read PusGeN faults: KEY=VALUE header block + 'lon,lat,value' lines; blocks are separated by blank lines."""
    segments = []
    if not os.path.exists(PUSGEN_FILE):
        return segments

    pts = []

    def flush():
        nonlocal pts
        if len(pts) > 1:
            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            segments.append({
                "pts": pts,
                "min_lon": min(lons), "max_lon": max(lons),
                "min_lat": min(lats), "max_lat": max(lats),
            })
        pts = []

    try:
        with open(PUSGEN_FILE, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or "=" in line:
                    flush()
                    continue
                parts = line.split(",")
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except (ValueError, IndexError):
                    continue
        flush()
    except Exception as e:
        print(f"Error loading PusGeN faults: {e}")
    return segments

pusgen_faults = load_pusgen_faults()
print(f"Loaded {len(pusgen_faults)} PusGeN fault segments.")

def normalize_lon(lon):
    """Keep longitude within [-180, 180] (prevents the +360 offset produced by Leaflet)"""
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return round(lon, 4)

# ==============================================================================
# ADMINISTRATOR LOGIN
# Credentials are read ONLY from server environment variables (Railway > Variables):
#   ADMIN_USERNAME, ADMIN_PASSWORD, AUTH_SECRET (a long random string)
# Never store credentials in the repository or in index.html.
# ==============================================================================
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
_AUTH_SECRET = (os.environ.get("AUTH_SECRET") or secrets.token_hex(32)).encode("utf-8")
if ADMIN_USERNAME and ADMIN_PASSWORD and not os.environ.get("AUTH_SECRET"):
    print("WARNING: AUTH_SECRET is not set; admin sessions are invalidated at every server restart.")

TOKEN_TTL_SECONDS = 8 * 3600     # admin session lifetime: 8 hours
MAX_FAILED_LOGINS = 5            # failed attempts per IP ...
LOCKOUT_SECONDS = 600            # ... within 10 minutes
_failed_logins = {}

ADMIN_FEATURES = [
    {
        "id": "insar_overlay_output",
        "title": "InSAR overlay on Generate Map",
        "description": "Overlay InSAR results (KMZ, GeoTIFF) on the PyGMT basemap, with a colorbar below the map.",
        "status": "active",
    },
    {
        "id": "volcano_source_deformation",
        "title": "Volcano source deformation database",
        "description": "Create and edit deformation source models (GNSS / InSAR) for volcanoes; they are shown to all users in the Volcano Deformation menu.",
        "status": "active",
    },
    {
        "id": "footprints_indonesia",
        "title": "Indonesia-wide footprints (Ascending / Descending)",
        "description": "Display all Sentinel-1 footprints covering Indonesia.",
        "status": "planned",
    },
    {
        "id": "data_availability",
        "title": "Data availability (ASF / ESA Copernicus)",
        "description": "Check or link data availability at ASF and Copernicus.",
        "status": "planned",
    },
    {
        "id": "arif_monitoring",
        "title": "ARIF Deformation Monitoring",
        "description": "ARIF deformation monitoring system (Advanced Research on InSAR and its Frontiers).",
        "status": "planned",
    },
]

def _b64e(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

def _b64d(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))

def make_token(username):
    payload = json.dumps({"u": username, "exp": int(time.time()) + TOKEN_TTL_SECONDS}).encode("utf-8")
    sig = hmac.new(_AUTH_SECRET, payload, "sha256").digest()
    return f"{_b64e(payload)}.{_b64e(sig)}"

def verify_token(token):
    """Return the payload dict if the token is valid and not expired; otherwise None."""
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload = _b64d(payload_b64)
        expected = hmac.new(_AUTH_SECRET, payload, "sha256").digest()
        if not hmac.compare_digest(expected, _b64d(sig_b64)):
            return None
        data = json.loads(payload.decode("utf-8"))
        if int(data.get("exp", 0)) < time.time():
            return None
        return data
    except Exception:
        return None

def get_admin_from_request():
    """Return the admin payload from the 'Authorization: Bearer <token>' header, or None."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return verify_token(header[7:].strip())

def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_admin_from_request() is None:
            return jsonify({"status": "error", "message": "Access denied: administrator login required."}), 401
        return fn(*args, **kwargs)
    return wrapper

def _client_ip():
    # The LAST X-Forwarded-For entry is appended by the trusted proxy (Railway); the first entry can be spoofed by the client.
    forwarded = [part.strip() for part in request.headers.get("X-Forwarded-For", "").split(",") if part.strip()]
    return forwarded[-1] if forwarded else (request.remote_addr or "unknown")

def _recent_failures(ip):
    now = time.time()
    attempts = [t for t in _failed_logins.get(ip, []) if now - t < LOCKOUT_SECONDS]
    _failed_logins[ip] = attempts
    return attempts

@app.route('/api/login', methods=['POST'])
def login():
    if not (ADMIN_USERNAME and ADMIN_PASSWORD):
        return jsonify({"status": "error", "message": "Administrator login is not configured on the server."}), 503

    ip = _client_ip()
    if len(_recent_failures(ip)) >= MAX_FAILED_LOGINS:
        return jsonify({"status": "error", "message": "Too many login attempts. Please try again in a few minutes."}), 429

    body = request.get_json(silent=True) or {}
    user_ok = hmac.compare_digest(str(body.get("username", "")).encode("utf-8"), ADMIN_USERNAME.encode("utf-8"))
    pass_ok = hmac.compare_digest(str(body.get("password", "")).encode("utf-8"), ADMIN_PASSWORD.encode("utf-8"))
    if user_ok and pass_ok:
        _failed_logins.pop(ip, None)
        return jsonify({
            "status": "success",
            "token": make_token(ADMIN_USERNAME),
            "username": ADMIN_USERNAME,
            "expires_in": TOKEN_TTL_SECONDS,
        })

    _failed_logins.setdefault(ip, []).append(time.time())
    return jsonify({"status": "error", "message": "Incorrect username or password."}), 401

@app.route('/api/admin/features', methods=['GET'])
@require_admin
def admin_features():
    return jsonify({"status": "success", "features": ADMIN_FEATURES})

# ==============================================================================
# VOLCANO DEFORMATION DATABASE (volcanoes + deformation source models)
# Stored in SQLite under DATA_DIR. On Railway, attach a Volume and set DATA_DIR to its mount path to keep the data
# across deployments. Without a Volume the database is re-created from volcano_deformation_seed.json after a redeploy.
# ==============================================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "volcano_deformation.db")
SEED_FILE = "volcano_deformation_seed.json"
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
KMZ_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.kmz$")
MAX_VOLCANOES = 500
MAX_SOURCES = 5000

SCHEMA = """
CREATE TABLE IF NOT EXISTS volcanoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    kmz TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    volcano_id INTEGER NOT NULL REFERENCES volcanoes(id) ON DELETE CASCADE,
    data_source TEXT NOT NULL CHECK (data_source IN ('GNSS', 'InSAR')),
    period_from TEXT NOT NULL,
    period_to TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    depth_km REAL NOT NULL,
    dvolume REAL NOT NULL,
    reference TEXT NOT NULL DEFAULT ''
);
"""

class DBUnavailable(Exception):
    pass

class ValidationError(ValueError):
    pass

@contextmanager
def db():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
    except (OSError, sqlite3.Error) as exc:
        raise DBUnavailable(str(exc))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def v_text(value, label, max_len, required=True):
    text = " ".join(str(value if value is not None else "").split())
    if required and not text:
        raise ValidationError(f"{label} is required.")
    if len(text) > max_len:
        raise ValidationError(f"{label} must be at most {max_len} characters.")
    return text

def v_number(value, label, lo, hi):
    if isinstance(value, bool):
        raise ValidationError(f"{label} must be a number.")
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{label} must be a number.")
    if not math.isfinite(num) or num < lo or num > hi:
        raise ValidationError(f"{label} must be between {lo:g} and {hi:g}.")
    return num

def v_month(value, label):
    text = str(value or "").strip()
    if not MONTH_RE.match(text):
        raise ValidationError(f"{label} must be a month in the format YYYY-MM.")
    return text

def validate_volcano(body):
    name = v_text(body.get("name"), "Volcano name", 80)
    lat = v_number(body.get("lat"), "Latitude", -90, 90)
    lon = v_number(body.get("lon"), "Longitude", -180, 180)
    kmz = str(body.get("kmz") or "").strip()
    if kmz and not KMZ_RE.match(kmz):
        raise ValidationError("The KMZ file name must look like 'ibu.kmz' (letters, digits, '_', '-', '.'; no folders).")
    return name, lat, lon, kmz

def validate_source(body):
    data_source = body.get("data_source")
    if data_source not in ("GNSS", "InSAR"):
        raise ValidationError("Data source must be GNSS or InSAR.")
    period_from = v_month(body.get("period_from"), "Period start")
    period_to = v_month(body.get("period_to"), "Period end")
    if period_from > period_to:
        raise ValidationError("The period start must not be later than the period end.")
    return {
        "data_source": data_source,
        "period_from": period_from,
        "period_to": period_to,
        "lat": v_number(body.get("lat"), "Source latitude", -90, 90),
        "lon": v_number(body.get("lon"), "Source longitude", -180, 180),
        "depth_km": v_number(body.get("depth_km"), "Source depth (km)", 0, 100),
        "dvolume": v_number(body.get("dvolume"), "Volume change (dV)", -1e6, 1e6),
        "reference": v_text(body.get("reference"), "Reference", 300, required=False),
    }

SOURCE_COLUMNS = ("id", "data_source", "period_from", "period_to", "lat", "lon", "depth_km", "dvolume", "reference")

def serialize_db(conn):
    by_volcano = {}
    rows = conn.execute(
        "SELECT id, volcano_id, data_source, period_from, period_to, lat, lon, depth_km, dvolume, reference "
        "FROM sources ORDER BY period_from DESC, id DESC").fetchall()
    for r in rows:
        by_volcano.setdefault(r["volcano_id"], []).append({k: r[k] for k in SOURCE_COLUMNS})
    out = []
    for r in conn.execute("SELECT id, name, lat, lon, kmz FROM volcanoes ORDER BY name COLLATE NOCASE").fetchall():
        out.append({"id": r["id"], "name": r["name"], "lat": r["lat"], "lon": r["lon"], "kmz": r["kmz"],
                    "sources": by_volcano.get(r["id"], [])})
    return out

def insert_source(conn, volcano_id, src):
    conn.execute(
        "INSERT INTO sources (volcano_id, data_source, period_from, period_to, lat, lon, depth_km, dvolume, reference) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (volcano_id, src["data_source"], src["period_from"], src["period_to"], src["lat"], src["lon"],
         src["depth_km"], src["dvolume"], src["reference"]))

def replace_all(conn, data):
    """Validate the whole payload first, then replace the database content (used by seeding and by import)."""
    if not isinstance(data, dict) or not isinstance(data.get("volcanoes"), list):
        raise ValidationError("Invalid file: a 'volcanoes' list is required.")
    volcanoes = data["volcanoes"]
    if len(volcanoes) > MAX_VOLCANOES:
        raise ValidationError(f"Too many volcanoes (maximum {MAX_VOLCANOES}).")
    parsed, names, total_sources = [], set(), 0
    for i, v in enumerate(volcanoes, start=1):
        if not isinstance(v, dict):
            raise ValidationError(f"Volcano #{i}: invalid entry.")
        try:
            name, lat, lon, kmz = validate_volcano(v)
            raw_sources = v.get("sources") or []
            if not isinstance(raw_sources, list):
                raise ValidationError("'sources' must be a list.")
            sources = [validate_source(x if isinstance(x, dict) else {}) for x in raw_sources]
        except ValidationError as exc:
            raise ValidationError(f"Volcano #{i}: {exc}")
        if name.lower() in names:
            raise ValidationError(f"Duplicate volcano name: {name}")
        names.add(name.lower())
        total_sources += len(sources)
        parsed.append((name, lat, lon, kmz, sources))
    if total_sources > MAX_SOURCES:
        raise ValidationError(f"Too many source models (maximum {MAX_SOURCES}).")
    conn.execute("DELETE FROM sources")
    conn.execute("DELETE FROM volcanoes")
    for name, lat, lon, kmz, sources in parsed:
        cur = conn.execute("INSERT INTO volcanoes (name, lat, lon, kmz) VALUES (?, ?, ?, ?)", (name, lat, lon, kmz))
        for src in sources:
            insert_source(conn, cur.lastrowid, src)

def init_db():
    """Create the tables; when the database is empty, load volcano_deformation_seed.json (if present)."""
    with db() as conn:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM volcanoes").fetchone()[0]
        if count == 0 and os.path.exists(SEED_FILE):
            with open(SEED_FILE, encoding="utf-8") as f:
                replace_all(conn, json.load(f))
            print(f"Volcano deformation database seeded from {SEED_FILE}.")
        total = conn.execute("SELECT COUNT(*) FROM volcanoes").fetchone()[0]
        print(f"Volcano deformation database ready: {total} volcanoes ({DB_PATH}).")

try:
    init_db()
except Exception as exc:   # the rest of the API must keep working if the database cannot be opened
    print(f"WARNING: the volcano deformation database could not be initialized: {exc}")

def db_error_response(exc):
    if isinstance(exc, ValidationError):
        return jsonify({"status": "error", "message": str(exc)}), 400
    if isinstance(exc, sqlite3.IntegrityError):
        return jsonify({"status": "error", "message": "A volcano with this name already exists."}), 409
    if isinstance(exc, DBUnavailable):
        return jsonify({"status": "error", "message": "The volcano database is unavailable on the server."}), 503
    raise exc

@app.route('/api/volcano-deformation', methods=['GET'])
def volcano_deformation_public():
    """Public: volcanoes with their deformation source models."""
    try:
        with db() as conn:
            return jsonify({"status": "success", "volcanoes": serialize_db(conn)})
    except DBUnavailable as exc:
        return db_error_response(exc)

@app.route('/api/admin/volcanoes', methods=['POST'])
@require_admin
def admin_save_volcano():
    body = request.get_json(silent=True) or {}
    try:
        name, lat, lon, kmz = validate_volcano(body)
        with db() as conn:
            vid = body.get("id")
            if vid is not None:
                cur = conn.execute("UPDATE volcanoes SET name=?, lat=?, lon=?, kmz=? WHERE id=?", (name, lat, lon, kmz, int(vid)))
                if cur.rowcount == 0:
                    return jsonify({"status": "error", "message": "Volcano not found."}), 404
            else:
                cur = conn.execute("INSERT INTO volcanoes (name, lat, lon, kmz) VALUES (?, ?, ?, ?)", (name, lat, lon, kmz))
                vid = cur.lastrowid
        return jsonify({"status": "success", "id": int(vid)})
    except (ValidationError, sqlite3.IntegrityError, DBUnavailable, ValueError, TypeError) as exc:
        if isinstance(exc, (ValueError, TypeError)) and not isinstance(exc, ValidationError):
            exc = ValidationError("Invalid volcano id.")
        return db_error_response(exc)

@app.route('/api/admin/volcanoes/<int:vid>', methods=['DELETE'])
@require_admin
def admin_delete_volcano(vid):
    try:
        with db() as conn:
            cur = conn.execute("DELETE FROM volcanoes WHERE id=?", (vid,))
            if cur.rowcount == 0:
                return jsonify({"status": "error", "message": "Volcano not found."}), 404
        return jsonify({"status": "success"})
    except DBUnavailable as exc:
        return db_error_response(exc)

@app.route('/api/admin/volcano-sources', methods=['POST'])
@require_admin
def admin_save_source():
    body = request.get_json(silent=True) or {}
    try:
        src = validate_source(body)
        with db() as conn:
            try:
                volcano_id = int(body.get("volcano_id"))
            except (TypeError, ValueError):
                raise ValidationError("Select a volcano first.")
            if conn.execute("SELECT 1 FROM volcanoes WHERE id=?", (volcano_id,)).fetchone() is None:
                return jsonify({"status": "error", "message": "Volcano not found."}), 404
            if body.get("id") is not None:
                cur = conn.execute(
                    "UPDATE sources SET data_source=?, period_from=?, period_to=?, lat=?, lon=?, depth_km=?, dvolume=?, reference=? "
                    "WHERE id=? AND volcano_id=?",
                    (src["data_source"], src["period_from"], src["period_to"], src["lat"], src["lon"], src["depth_km"],
                     src["dvolume"], src["reference"], int(body["id"]), volcano_id))
                if cur.rowcount == 0:
                    return jsonify({"status": "error", "message": "Source model not found."}), 404
                sid = int(body["id"])
            else:
                if conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] >= MAX_SOURCES:
                    raise ValidationError(f"Too many source models (maximum {MAX_SOURCES}).")
                insert_source(conn, volcano_id, src)
                sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return jsonify({"status": "success", "id": sid})
    except (ValidationError, DBUnavailable) as exc:
        return db_error_response(exc)

@app.route('/api/admin/volcano-sources/<int:sid>', methods=['DELETE'])
@require_admin
def admin_delete_source(sid):
    try:
        with db() as conn:
            cur = conn.execute("DELETE FROM sources WHERE id=?", (sid,))
            if cur.rowcount == 0:
                return jsonify({"status": "error", "message": "Source model not found."}), 404
        return jsonify({"status": "success"})
    except DBUnavailable as exc:
        return db_error_response(exc)

@app.route('/api/admin/volcano-deformation/export', methods=['GET'])
@require_admin
def admin_export_volcano_db():
    """Download the database as JSON (without internal ids); it can be re-imported or committed as the seed file."""
    try:
        with db() as conn:
            volcanoes = [
                {"name": v["name"], "lat": v["lat"], "lon": v["lon"], "kmz": v["kmz"],
                 "sources": [{k: x[k] for k in SOURCE_COLUMNS if k != "id"} for x in v["sources"]]}
                for v in serialize_db(conn)
            ]
        return jsonify({"version": 1, "data_dir": DATA_DIR, "volcanoes": volcanoes})
    except DBUnavailable as exc:
        return db_error_response(exc)

@app.route('/api/admin/volcano-deformation/import', methods=['POST'])
@require_admin
def admin_import_volcano_db():
    """Replace the whole database with the uploaded JSON (validated first; nothing changes if validation fails)."""
    try:
        with db() as conn:
            replace_all(conn, request.get_json(silent=True))
            n_v = conn.execute("SELECT COUNT(*) FROM volcanoes").fetchone()[0]
            n_s = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        return jsonify({"status": "success", "volcanoes": n_v, "sources": n_s})
    except (ValidationError, DBUnavailable) as exc:
        return db_error_response(exc)

# ==============================================================================
# ELEVATION CONTOURS FOR THE INTERACTIVE MAP (200 m interval, land only)
# ==============================================================================
CONTOUR_MAX_SPAN = 5.0   # degrees; must match the limit in index.html

@app.route('/api/contours', methods=['POST'])
def contours():
    if contourpy is None:
        return jsonify({"status": "error", "message": "The contourpy package is not installed on the server (add it to the Dockerfile)."}), 503
    try:
        body = request.get_json(silent=True) or {}
        min_lon = normalize_lon(float(body["minLon"]))
        max_lon = normalize_lon(float(body["maxLon"]))
        min_lat = float(body["minLat"])
        max_lat = float(body["maxLat"])
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon
        if max_lon - min_lon > CONTOUR_MAX_SPAN or max_lat - min_lat > CONTOUR_MAX_SPAN:
            return jsonify({"status": "error", "message": f"Region too large (maximum {CONTOUR_MAX_SPAN:g}° x {CONTOUR_MAX_SPAN:g}°)."}), 400
        if max_lat <= min_lat:
            return jsonify({"status": "error", "message": "Invalid region."}), 400

        resolution = "15s"
        grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=[min_lon, max_lon, min_lat, max_lat])
        z = np.asarray(grid.values, dtype=float)
        x = np.asarray(grid["lon"].values, dtype=float)
        y = np.asarray(grid["lat"].values, dtype=float)
        z = np.where(np.isfinite(z), z, -9999.0)

        z_max = float(z.max())
        features = []
        if z_max >= 200:
            gen = contourpy.contour_generator(x, y, z, line_type=contourpy.LineType.Separate)
            for level in np.arange(200, z_max + 1, 200):
                lines = gen.lines(float(level))
                coords = [np.round(np.asarray(ln), 4).tolist() for ln in lines if len(ln) >= 2]
                if coords:
                    features.append({
                        "type": "Feature",
                        "properties": {"level": int(level)},
                        "geometry": {"type": "MultiLineString", "coordinates": coords},
                    })
        return jsonify({
            "status": "success",
            "resolution": resolution,
            "levels": len(features),
            "geojson": {"type": "FeatureCollection", "features": features},
        })
    except (KeyError, TypeError, ValueError):
        return jsonify({"status": "error", "message": "Region parameters are missing or invalid."}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================================
# INSAR OVERLAY (administrators only): KMZ/MintPy/GeoTIFF -> image on the PyGMT map
# ==============================================================================
OVERLAY_KEY_RGB = (253, 1, 252)   # key color for transparent pixels (GMT image: -G<color>+t)
ALLOWED_CMAPS = {"jet", "viridis", "turbo", "roma", "vik", "polar"}

# API version: incremented whenever a new server feature is added; index.html checks it through GET /
API_VERSION = 4

def _mercator_y(lat_deg):
    return math.log(math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0))

def prepare_overlay_png(image_b64, west, south, east, north, region, map_width_cm=18.0):
    """Crop the image to the map region, downsample if needed, and replace transparent pixels with the key color.
    Return (png_path, west, south, east, north, width_cm, height_cm, RGBA_image), or None if the image does not intersect the region."""
    min_lon, max_lon, min_lat, max_lat = region
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGBA")
    width, height = img.size
    if east <= west or north <= south or width < 2 or height < 2:
        return None

    ix_w, ix_e = max(west, min_lon), min(east, max_lon)
    ix_s, ix_n = max(south, min_lat), min(north, max_lat)
    if ix_w >= ix_e or ix_s >= ix_n:
        return None

    left = max(int(math.floor((ix_w - west) / (east - west) * width)), 0)
    right = min(int(math.ceil((ix_e - west) / (east - west) * width)), width)
    top = max(int(math.floor((north - ix_n) / (north - south) * height)), 0)
    bottom = min(int(math.ceil((north - ix_s) / (north - south) * height)), height)
    if right - left < 1 or bottom - top < 1:
        return None
    img = img.crop((left, top, right, bottom))

    # Actual geographic extent after rounding to whole pixels
    w2 = west + left / width * (east - west)
    e2 = west + right / width * (east - west)
    n2 = north - top / height * (north - south)
    s2 = north - bottom / height * (north - south)

    # Size on paper (Mercator projection, M18c)
    lon_span_rad = math.radians(max_lon - min_lon)
    w_cm = map_width_cm * math.radians(e2 - w2) / lon_span_rad
    h_cm = map_width_cm * (_mercator_y(n2) - _mercator_y(s2)) / lon_span_rad

    # Sufficient resolution: 300 dpi x 2, longest side at least 600 pixels
    target = max(int(max(w_cm, h_cm) / 2.54 * 300 * 2), 600)
    if max(img.size) > target:
        scale = target / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.NEAREST)

    arr = np.array(img)
    transparent = arr[:, :, 3] < 128
    rgb = arr[:, :, :3].copy()
    key = np.array(OVERLAY_KEY_RGB, dtype=rgb.dtype)
    clash = (~transparent) & np.all(rgb == key, axis=2)       # data pixels that happen to equal the key color
    rgb[clash] = (key[0] - 2, key[1], key[2])
    rgb[transparent] = key

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    Image.fromarray(rgb, "RGB").save(path, "PNG")
    return path, w2, s2, e2, n2, w_cm, h_cm, img

def draw_insar_overlays(fig, region, overlays, tmp_files, warnings):
    """Draw all GroundOverlay images (raster KMZ / RGB GeoTIFF) on top of the basemap. Return the number drawn."""
    drawn = 0
    key_spec = "{}/{}/{}+t".format(*OVERLAY_KEY_RGB)
    for idx, ov in enumerate(overlays, start=1):
        try:
            prep = prepare_overlay_png(
                str(ov["image_base64"]),
                float(ov["west"]), float(ov["south"]), float(ov["east"]), float(ov["north"]),
                region,
            )
        except Exception as exc:
            warnings.append(f"InSAR overlay #{idx}: the image could not be processed ({exc}).")
            continue
        if prep is None:
            warnings.append(f"InSAR overlay #{idx}: it does not intersect the map region. Set the region to the extent of the InSAR result.")
            continue
        path, w, s, e, n, w_cm, h_cm, rgba = prep
        tmp_files.append(path)
        position = f"g{w:.6f}/{s:.6f}+w{w_cm:.4f}c/{h_cm:.4f}c+jBL"
        try:
            # The -G<color>+t option (transparent color) has no alias in PyGMT, so it is passed as G=
            fig.image(imagefile=path, position=position, G=key_spec)
            drawn += 1
            continue
        except Exception as exc_key:
            first_error = exc_key
        try:
            # Fallback: the RGBA PNG as is (transparency depends on GMT alpha-channel support)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                rgba_path = tmp.name
            tmp_files.append(rgba_path)
            rgba.save(rgba_path, "PNG")
            fig.image(imagefile=rgba_path, position=position)
            warnings.append(f"InSAR overlay #{idx}: the transparent-color option was rejected ({first_error}); the data edges may not be transparent.")
            drawn += 1
        except Exception as exc:
            warnings.append(f"InSAR overlay #{idx} could not be drawn by PyGMT: {exc}")
    return drawn

def make_insar_cpt(spec, tmp_files, warnings):
    """Create a CPT file from a colormap and a value range. Return the CPT path, or None if the parameters are invalid."""
    try:
        vmin = float(spec["vmin"])
        vmax = float(spec["vmax"])
    except (KeyError, TypeError, ValueError):
        warnings.append("InSAR colorbar not created: Min/Max values are missing or invalid.")
        return None
    if not (math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin):
        warnings.append("InSAR colorbar not created: Max must be greater than Min.")
        return None
    cmap = str(spec.get("cmap", "jet"))
    if cmap not in ALLOWED_CMAPS:
        cmap = "jet"
    with tempfile.NamedTemporaryFile(suffix=".cpt", delete=False) as tmp:
        path = tmp.name
    tmp_files.append(path)
    try:
        pygmt.makecpt(cmap=cmap, series=[vmin, vmax], reverse=bool(spec.get("reverse", False)), continuous=True, output=path)
    except Exception as exc:
        warnings.append(f"The InSAR color palette (CPT) could not be created: {exc}")
        return None
    return path

GEOTIFF_MAX_BYTES = 40 * 1024 * 1024

def draw_insar_geotiff(fig, region, proj, spec, cpt_path, tmp_files, warnings):
    """Render a single-band GeoTIFF (EPSG:4326) directly with GMT/GDAL using the selected colormap and value range."""
    if cpt_path is None:
        warnings.append("GeoTIFF not drawn: the colormap or value range is invalid.")
        return 0
    b64 = str(spec.get("geotiff_base64") or "")
    if not b64:
        warnings.append("GeoTIFF not drawn: the file data was not received by the server.")
        return 0
    try:
        raw = base64.b64decode(b64)
    except Exception:
        warnings.append("GeoTIFF not drawn: the file data is corrupted.")
        return 0
    if len(raw) > GEOTIFF_MAX_BYTES:
        warnings.append(f"GeoTIFF not drawn: the file exceeds {GEOTIFF_MAX_BYTES // (1024 * 1024)} MB.")
        return 0
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    tmp_files.append(path)
    last_error = None
    for grid_arg in (path + "=gd", path):   # "=gd" = read through GDAL
        try:
            fig.grdimage(grid=grid_arg, region=region, projection=proj, cmap=cpt_path, nan_transparent=True)
            return 1
        except Exception as exc:
            last_error = exc
    warnings.append(f"The GeoTIFF could not be drawn by PyGMT/GDAL: {last_error}")
    return 0

def draw_insar_points(fig, region, spec, cpt_path, warnings):
    """Color time-series points (MintPy KMZ) by mean velocity, using the same CPT as the colorbar."""
    if cpt_path is None:
        warnings.append("InSAR points not drawn: the colormap or value range is invalid.")
        return 0
    min_lon, max_lon, min_lat, max_lat = region
    xs, ys, vs = [], [], []
    for pt in spec.get("points", []) or []:
        try:
            lon = normalize_lon(float(pt["lon"]))
            lat = float(pt["lat"])
            val = float(pt["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(val) and min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            xs.append(lon)
            ys.append(lat)
            vs.append(val)
    if not xs:
        warnings.append("No InSAR points fall inside the map region. Set the region to the extent of the InSAR result.")
        return 0
    try:
        size = min(max(float(spec.get("pointSize", 6)), 1.0), 30.0)
    except (TypeError, ValueError):
        size = 6.0
    fig.plot(x=xs, y=ys, style=f"c{size:g}p", fill=vs, cmap=cpt_path, pen="0.4p,black")

    ref = spec.get("refpoint")
    if isinstance(ref, dict):
        try:
            ref_lon = normalize_lon(float(ref["lon"]))
            ref_lat = float(ref["lat"])
            color = str(ref.get("color", ""))
            if not (len(color) == 7 and color[0] == "#" and all(ch in "0123456789abcdefABCDEF" for ch in color[1:])):
                color = "#77ff80"
            if min_lon <= ref_lon <= max_lon and min_lat <= ref_lat <= max_lat:
                fig.plot(x=[ref_lon], y=[ref_lat], style=f"a{size * 2:g}p", fill=color, pen="0.8p,black")
        except (KeyError, TypeError, ValueError):
            pass
    return 1

def draw_insar_layers(fig, region, proj, spec, cpt_path, tmp_files, warnings):
    """Choose the drawing method by data type: raster (image), geotiff, or points. Return the number of layers drawn."""
    kind = spec.get("kind") or ("raster" if spec.get("images") else "")
    if kind in ("raster", "images"):
        return draw_insar_overlays(fig, region, spec.get("images", []) or [], tmp_files, warnings)
    if kind == "geotiff":
        return draw_insar_geotiff(fig, region, proj, spec, cpt_path, tmp_files, warnings)
    if kind == "points":
        return draw_insar_points(fig, region, spec, cpt_path, warnings)
    warnings.append("Unrecognized InSAR data type; the overlay was ignored.")
    return 0

def draw_insar_colorbar(fig, spec, cpt_path, cbar_spec, warnings):
    """Native GMT InSAR colorbar (below or to the right of the map)."""
    forbidden = set('+"\'\\@|')   # characters with special meaning in GMT (the '+' modifier, quotes, backslash, '@' escape, '|')
    label = "".join(ch for ch in str(spec.get("label", "")) if ch not in forbidden)[:80].strip()
    with pygmt.config(FORMAT_FLOAT_MAP="%g"):
        try:
            frame = ["a", f"x+l{label}"] if label else ["a"]
            fig.colorbar(cmap=cpt_path, position=cbar_spec, box=False, frame=frame)
        except Exception as exc:
            warnings.append(f"The InSAR colorbar label could not be used ({exc}); the colorbar was drawn without a label.")
            fig.colorbar(cmap=cpt_path, position=cbar_spec, box=False, frame=["a"])
    return True

def volcano_db_status():
    try:
        with db() as conn:
            return {"available": True, "volcanoes": conn.execute("SELECT COUNT(*) FROM volcanoes").fetchone()[0]}
    except Exception:
        return {"available": False, "volcanoes": 0}

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "service": "PyGMT Cartographic API",
        "api_version": API_VERSION,
        "volcanoes_loaded": len(df_volcanoes),
        "pusgen_segments_loaded": len(pusgen_faults),
        "admin_login_configured": bool(ADMIN_USERNAME and ADMIN_PASSWORD),
        "contours_available": contourpy is not None,
        "insar_overlay_available": Image is not None,
        "volcano_database": volcano_db_status()
    }), 200
@app.route('/api/generate-map', methods=['POST'])
def generate_map():
    tmp_files = []  # temporary files removed at the end of the request
    warnings = []   # notes for the user (layers that were requested but could not be drawn)
    try:
        data = request.json
        admin_user = get_admin_from_request()   # None if the caller is not an administrator
        
        # 1. Coordinate parameters with longitude normalization
        raw_min_lon = float(data.get('minLon', 121.5))
        raw_max_lon = float(data.get('maxLon', 124.5))
        min_lat = float(data.get('minLat', -9.2))
        max_lat = float(data.get('maxLat', -7.6))

        min_lon = normalize_lon(raw_min_lon)
        max_lon = normalize_lon(raw_max_lon)
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon

        region = [min_lon, max_lon, min_lat, max_lat]
        
        # 2. Cartographic parameters
        cpt = data.get('cptType', 'topography')
        cbar_pos = data.get('cbarPos', 'bottom')
        frame_style = data.get('frameStyle', 'plain')
        
        # Scalebar
        show_scalebar = data.get('showScalebar', False)
        scale_km = int(data.get('scalebarKm', 50))
        
        # Geophysical layers
        show_faults = data.get('showFaults', False)
        show_volcanoes = data.get('showVolcanoes', False)
        show_contours = data.get('showContours', False)
        show_pusgen = data.get('showPusgen', False)
        client_pusgen = data.get('pusgenSegments')   # [[[lon, lat], ...], ...] from the browser (independent of the server file)

        # Earthquakes M>=5 (USGS data loaded in the browser and sent here)
        show_eq = data.get('showEq', False)
        eq_events = data.get('eqEvents', []) or []

        # InSAR overlay + colorbar (administrators only)
        insar_req = data.get('insarOverlay')
        use_insar = False
        if insar_req:
            if admin_user is None:
                warnings.append("The InSAR overlay is only available to administrators; this option was ignored.")
            elif Image is None:
                warnings.append("InSAR overlay not drawn: the Pillow package is not installed on the server (add it to the Dockerfile).")
            else:
                use_insar = True

        # GNSS stations (format sent by the frontend: [{"id": "bako", "lat": -6.309, "lon": 106.203}, ...])
        show_gps = data.get('showGps', False)
        show_gps_labels = bool(data.get('gpsLabels', False))
        gps_stations = data.get('gpsStations', []) or []
        try:
            gps_size = float(data.get('gpsSize', 12))
        except (TypeError, ValueError):
            gps_size = 12.0
        gps_size = min(max(gps_size, 1.0), 60.0)
        
        # InSAR Footprints
        show_asc = data.get('showAsc', False)
        asc_coords = data.get('ascCoords', {})
        asc_frame = data.get('ascFrame', '')

        show_dsc = data.get('showDsc', False)
        dsc_coords = data.get('dscCoords', {})
        dsc_frame = data.get('dscFrame', '')

        # Initialize PyGMT
        fig = pygmt.Figure()
        proj = "M18c"
        
        # GMT configuration:
        # FORMAT_GEO_MAP="ddd.xx": at most two decimals, minus sign for the Southern Hemisphere
        # FORMAT_FLOAT_MAP="%.0f": colorbar annotations as whole numbers (integers)
        pygmt.config(
            FORMAT_GEO_MAP="ddd.xx",
            FORMAT_FLOAT_MAP="%.0f"
        )

        # Clean frame without grid lines
        frame_code = "f" if frame_style == "zebra" else "p"
        fig_frame = ["a", frame_code]

        # A. Load Earth Relief (adaptive high resolution)
        lon_span = abs(max_lon - min_lon)
        lat_span = abs(max_lat - min_lat)
        if lon_span <= 1.5 and lat_span <= 1.5:
            res = "01s"  # detailed resolution, ~30 m
        elif lon_span <= 4.5 and lat_span <= 3.5:
            res = "03s"
        else:
            res = "15s"

        grid = pygmt.datasets.load_earth_relief(resolution=res, region=region)
        cmap_name = "geo" if cpt == "terrain" else ("bathy" if cpt == "bathymetry" else "relief")

        is_hillshade = (cpt == "hillshade")
        cpt_path = None

        if is_hillshade:
            # Light-gray hillshade for land only (the ocean is left transparent/empty)
            # Hillshade intensity (illumination from the northwest, azimuth 315)
            shade = pygmt.grdgradient(grid=grid, azimuth=315, normalize="t1")
            # Mask the ocean (elevation < 0 m)
            shade_land = shade.where(grid.values >= 0)

            # Gray CPT: intensity 0 -> light gray (~194/255), sunlit slopes -> white, shaded slopes -> dark gray
            with tempfile.NamedTemporaryFile(mode="w", suffix=".cpt", delete=False) as tmp_cpt:
                tmp_cpt.write("-3.2\t0\t1.0\t255\nB\t0\nF\t255\nN\t255\n")
                cpt_path = tmp_cpt.name
            tmp_files.append(cpt_path)

            fig.grdimage(
                grid=shade_land,
                region=region,
                projection=proj,
                cmap=cpt_path,
                nan_transparent=True,
                frame=fig_frame
            )
        else:
            fig.grdimage(
                grid=grid,
                region=region,
                projection=proj,
                cmap=cmap_name,
                shading=True,
                frame=fig_frame
            )

        # A2. InSAR results (administrators) on top of the basemap, below contours, shorelines, and other layers
        insar_drawn = 0
        insar_cpt = None
        if use_insar:
            insar_cpt = make_insar_cpt(insar_req, tmp_files, warnings)
            insar_drawn = draw_insar_layers(fig, region, proj, insar_req, insar_cpt, tmp_files, warnings)

       # B. Elevation contours: 200 m interval, integer annotations, land only (>= 200 m)
        if show_contours:
            fig.grdcontour(
                grid=grid,
                interval=200,
                annotation="200+f7p,Helvetica-Bold,gray20",
                limit=[200, 9000],  # Uses the -L parameter to restrict the elevation range to land
                pen="0.35p,gray40"
            )

        # Shorelines
        fig.coast(
            region=region,
            projection=proj,
            shorelines="0.6p,gray20",
            resolution="f"
        )

        # C. InSAR Footprints
        if show_asc and asc_coords:
            ul, ur, lr, ll = asc_coords.get('ul'), asc_coords.get('ur'), asc_coords.get('lr'), asc_coords.get('ll')
            asc_x = [normalize_lon(ul[0]), normalize_lon(ur[0]), normalize_lon(lr[0]), normalize_lon(ll[0]), normalize_lon(ul[0])]
            asc_y = [ul[1], ur[1], lr[1], ll[1], ul[1]]
            fig.plot(x=asc_x, y=asc_y, pen="1.8p,red", fill="red@80")
            fig.text(x=float(np.mean(asc_x[:4])), y=float(np.mean(asc_y[:4])) + 0.04, text=f"Asc: {asc_frame}", font="9p,Helvetica-Bold,red3")

        if show_dsc and dsc_coords:
            ul, ur, lr, ll = dsc_coords.get('ul'), dsc_coords.get('ur'), dsc_coords.get('lr'), dsc_coords.get('ll')
            dsc_x = [normalize_lon(ul[0]), normalize_lon(ur[0]), normalize_lon(lr[0]), normalize_lon(ll[0]), normalize_lon(ul[0])]
            dsc_y = [ul[1], ur[1], lr[1], ll[1], ul[1]]
            fig.plot(x=dsc_x, y=dsc_y, pen="1.8p,blue", fill="blue@80")
            fig.text(x=float(np.mean(dsc_x[:4])), y=float(np.mean(dsc_y[:4])) - 0.04, text=f"Dsc: {dsc_frame}", font="9p,Helvetica-Bold,blue3")

        # D. Active faults (Bird, 2003)
        if show_faults:
            try:
                fig.plot(
                    data=FAULT_FILE if os.path.exists(FAULT_FILE) else FAULT_URL,
                    pen="1.8p,red3"
                )
            except Exception:
                pass

        # D2. PusGeN 2024 faults: prefer the geometry sent by the browser; fallback: the file on the server
        if show_pusgen:
            seg_lists = []
            source = ""
            if isinstance(client_pusgen, list) and client_pusgen:
                source = "browser"
                for seg in client_pusgen:
                    try:
                        pts = [(float(pt[0]), float(pt[1])) for pt in seg]
                    except (TypeError, ValueError, IndexError):
                        continue
                    if len(pts) > 1:
                        seg_lists.append(pts)
            elif pusgen_faults:
                source = "server"
                seg_lists = [
                    sg["pts"] for sg in pusgen_faults
                    if sg["max_lon"] >= min_lon and sg["min_lon"] <= max_lon
                    and sg["max_lat"] >= min_lat and sg["min_lat"] <= max_lat
                ]
            else:
                warnings.append("PusGeN layer not drawn: pusgen_fault_2024.txt is missing on the server and the browser did not send fault geometry.")

            if seg_lists:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_f:
                    for pts in seg_lists:
                        tmp_f.write(">\n")
                        for lon, lat in pts:
                            tmp_f.write(f"{lon:.5f} {lat:.5f}\n")
                    pusgen_path = tmp_f.name
                tmp_files.append(pusgen_path)
                fig.plot(data=pusgen_path, pen="1p,magenta3")
                print(f"PusGeN: {len(seg_lists)} segments drawn (source: {source}).")
            elif source:
                warnings.append("No PusGeN fault segments fall inside this map region.")

        # D3. Earthquakes M>=5 (USGS): circles scaled with magnitude
        if show_eq and eq_events:
            quakes = []
            for ev in eq_events:
                try:
                    ev_lon = normalize_lon(float(ev["lon"]))
                    ev_lat = float(ev["lat"])
                    ev_mag = float(ev["mag"])
                except (KeyError, TypeError, ValueError):
                    continue
                if min_lon <= ev_lon <= max_lon and min_lat <= ev_lat <= max_lat:
                    quakes.append((ev_lon, ev_lat, ev_mag))
            if quakes:
                quakes.sort(key=lambda q: -q[2])   # largest first so that smaller events stay on top
                fig.plot(
                    x=[q[0] for q in quakes],
                    y=[q[1] for q in quakes],
                    size=[0.08 * max(q[2] - 3.0, 0.5) ** 2 for q in quakes],   # diameter (cm)
                    style="cc",
                    fill="255/165/0@50",
                    pen="0.4p,black"
                )
            else:
                warnings.append("No M>=5 earthquakes fall inside the map region for the selected period.")
        elif show_eq:
            warnings.append("Earthquakes not drawn: the earthquake data has not been loaded in the browser.")

        # E. Holocene volcanoes (GVP database)
        if show_volcanoes and df_volcanoes.empty:
            warnings.append("Volcano layer not drawn: the GVP database could not be read on the server.")
        if show_volcanoes and not df_volcanoes.empty:
            mask = (
                (df_volcanoes['Longitude'] >= min_lon) & 
                (df_volcanoes['Longitude'] <= max_lon) & 
                (df_volcanoes['Latitude'] >= min_lat) & 
                (df_volcanoes['Latitude'] <= max_lat)
            )
            v_in_region = df_volcanoes[mask]
            if not v_in_region.empty:
                fig.plot(
                    x=v_in_region['Longitude'].values,
                    y=v_in_region['Latitude'].values,
                    style="t0.35c",
                    fill="red3",
                    pen="0.6p,black"
                )

        # E2. GNSS stations: blue circles (size in points)
        if show_gps and gps_stations:
            gps_lon, gps_lat, gps_ids = [], [], []
            for st in gps_stations:
                try:
                    st_lat = float(st['lat'])
                    st_lon = normalize_lon(float(st['lon']))
                except (KeyError, TypeError, ValueError):
                    continue
                if min_lon <= st_lon <= max_lon and min_lat <= st_lat <= max_lat:
                    gps_lon.append(st_lon)
                    gps_lat.append(st_lat)
                    # '@' and '\\' have a special meaning in GMT text
                    gps_ids.append(str(st.get('id', '')).replace('\\', '').replace('@', '@@'))
            if gps_lon:
                fig.plot(
                    x=gps_lon,
                    y=gps_lat,
                    style=f"c{gps_size:g}p",
                    fill="blue",
                    pen="0.6p,black"
                )
                # Station ID directly above the symbol (bottom of the text = 2 pt above the circle)
                if show_gps_labels:
                    fig.text(
                        x=gps_lon,
                        y=gps_lat,
                        text=gps_ids,
                        justify="BC",
                        offset=f"0/{gps_size / 2 + 2:g}p",
                        font="8p,Helvetica-Bold,black",
                        fill="white@30",
                        clearance="0.5p/0.5p"
                    )
            else:
                warnings.append("No GNSS stations fall inside the map region.")

        # F. Scale bar in the southwest (lower-left) corner with 60% transparency
        if show_scalebar:
            fig.basemap(
                map_scale=f"x0.6c/0.6c+c{min_lat}+w{scale_km}k+lkm+f+jBL",
                box="+gwhite@40+p0.5p,gray30+r2p"
            )

        # G. Colorbar: InSAR (if an overlay is active) or elevation (except in hillshade mode)
        cbar_spec = "JMR+w8c/0.5c+o1.0c/0c" if cbar_pos == "right" else "JBC+w10c/0.5c+o0c/1.4c+h"
        if use_insar and insar_drawn > 0 and insar_cpt:
            draw_insar_colorbar(fig, insar_req, insar_cpt, cbar_spec, warnings)
        elif not is_hillshade and not (use_insar and insar_drawn > 0):
            fig.colorbar(
                position=cbar_spec,
                box=False,
                frame=["a1000", "x+lElevation", "y+lm"]
            )

        # H. Save the figure at 300 DPI
        # Transparent PNG (background outside the map is transparent); the JPEG is created in the frontend on a white background
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            output_png = tmp_png.name
        tmp_files.append(output_png)
        fig.savefig(output_png, dpi=300, transparent=True)

        with open(output_png, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "status": "success",
            "image_base64": f"data:image/png;base64,{encoded_img}",
            "warnings": warnings,
            "admin": admin_user is not None   # lets the frontend detect an expired admin session
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        for tmp in tmp_files:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
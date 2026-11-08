import os
import io
import math
import hmac
import json
import time
import secrets
import tempfile
import base64
from functools import wraps
import pandas as pd
import numpy as np
import pygmt
from flask import Flask, request, jsonify
from flask_cors import CORS

# Paket opsional: tanpa keduanya server tetap jalan, hanya fitur terkait yang dinonaktifkan
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

# Database Gunung Api GVP
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
    """Baca patahan PusGeN: blok header KEY=VALUE + baris 'lon,lat,nilai', dipisah baris kosong."""
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
    """Menjamin bujur selalu berada dalam rentang [-180, 180] (mencegah bug +360 dari Leaflet)"""
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return round(lon, 4)

# ==============================================================================
# ADMINISTRATOR LOGIN
# Kredensial HANYA dibaca dari environment variable server (Railway > Variables):
#   ADMIN_USERNAME, ADMIN_PASSWORD, AUTH_SECRET (string acak panjang)
# Jangan menaruh kredensial di repo atau di index.html.
# ==============================================================================
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
_AUTH_SECRET = (os.environ.get("AUTH_SECRET") or secrets.token_hex(32)).encode("utf-8")
if ADMIN_USERNAME and ADMIN_PASSWORD and not os.environ.get("AUTH_SECRET"):
    print("PERINGATAN: AUTH_SECRET belum diset; sesi admin akan hilang setiap server restart.")

TOKEN_TTL_SECONDS = 8 * 3600     # sesi admin berlaku 8 jam
MAX_FAILED_LOGINS = 5            # percobaan gagal per IP ...
LOCKOUT_SECONDS = 600            # ... dalam 10 menit
_failed_logins = {}

ADMIN_FEATURES = [
    {
        "id": "insar_overlay_output",
        "title": "InSAR overlay pada Generate Map",
        "description": "Hasil InSAR (KMZ) ditumpuk di atas basemap pada peta PyGMT, lengkap dengan colorbar di bawah peta.",
        "status": "active",
    },
    {
        "id": "footprints_indonesia",
        "title": "Footprint seluruh Indonesia (Ascending / Descending)",
        "description": "Menampilkan semua footprint Sentinel-1 yang mencakup Indonesia.",
        "status": "planned",
    },
    {
        "id": "data_availability",
        "title": "Ketersediaan data (ASF Alaska / ESA Copernicus)",
        "description": "Mengecek atau menautkan ketersediaan data ke ASF dan Copernicus.",
        "status": "planned",
    },
    {
        "id": "arif_monitoring",
        "title": "ARIF Deformation Monitoring",
        "description": "Sistem monitoring deformasi ARIF (Advanced Research on InSAR and its Frontiers).",
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
    """Return payload dict jika token sah dan belum kedaluwarsa, selain itu None."""
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
    """Return payload admin dari header 'Authorization: Bearer <token>', atau None."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return verify_token(header[7:].strip())

def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_admin_from_request() is None:
            return jsonify({"status": "error", "message": "Akses ditolak: login administrator diperlukan."}), 401
        return fn(*args, **kwargs)
    return wrapper

def _client_ip():
    # Entri TERAKHIR X-Forwarded-For ditambahkan oleh proxy tepercaya (Railway); entri pertama bisa dipalsukan klien.
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
        return jsonify({"status": "error", "message": "Login administrator belum dikonfigurasi di server."}), 503

    ip = _client_ip()
    if len(_recent_failures(ip)) >= MAX_FAILED_LOGINS:
        return jsonify({"status": "error", "message": "Terlalu banyak percobaan login. Coba lagi dalam beberapa menit."}), 429

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
    return jsonify({"status": "error", "message": "Username atau password salah."}), 401

@app.route('/api/admin/features', methods=['GET'])
@require_admin
def admin_features():
    return jsonify({"status": "success", "features": ADMIN_FEATURES})

# ==============================================================================
# KONTUR ELEVASI UNTUK PETA INTERAKTIF (interval 200 m, daratan)
# ==============================================================================
CONTOUR_MAX_SPAN = 5.0   # derajat; harus sama dengan batas di index.html

@app.route('/api/contours', methods=['POST'])
def contours():
    if contourpy is None:
        return jsonify({"status": "error", "message": "Paket contourpy belum terpasang di server (tambahkan ke Dockerfile)."}), 503
    try:
        body = request.get_json(silent=True) or {}
        min_lon = normalize_lon(float(body["minLon"]))
        max_lon = normalize_lon(float(body["maxLon"]))
        min_lat = float(body["minLat"])
        max_lat = float(body["maxLat"])
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon
        if max_lon - min_lon > CONTOUR_MAX_SPAN or max_lat - min_lat > CONTOUR_MAX_SPAN:
            return jsonify({"status": "error", "message": f"Region terlalu luas (maks {CONTOUR_MAX_SPAN:g}° x {CONTOUR_MAX_SPAN:g}°)."}), 400
        if max_lat <= min_lat:
            return jsonify({"status": "error", "message": "Region tidak valid."}), 400

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
        return jsonify({"status": "error", "message": "Parameter region tidak lengkap atau tidak valid."}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ==============================================================================
# OVERLAY INSAR (khusus administrator): KMZ/MintPy -> gambar pada peta PyGMT
# ==============================================================================
OVERLAY_KEY_RGB = (253, 1, 252)   # warna kunci untuk piksel transparan (GMT image: -G<warna>+t)
ALLOWED_CMAPS = {"jet", "viridis", "turbo", "roma", "vik", "polar"}

def _mercator_y(lat_deg):
    return math.log(math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0))

def prepare_overlay_png(image_b64, west, south, east, north, region, map_width_cm=18.0):
    """Potong gambar sesuai region peta, kecilkan seperlunya, ganti piksel transparan dengan warna kunci.
    Return (path_png, west, south, east, north, lebar_cm, tinggi_cm) atau None bila tidak beririsan dengan region."""
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

    # Batas geografis aktual setelah pembulatan ke piksel
    w2 = west + left / width * (east - west)
    e2 = west + right / width * (east - west)
    n2 = north - top / height * (north - south)
    s2 = north - bottom / height * (north - south)

    # Ukuran di kertas (proyeksi Mercator M18c)
    lon_span_rad = math.radians(max_lon - min_lon)
    w_cm = map_width_cm * math.radians(e2 - w2) / lon_span_rad
    h_cm = map_width_cm * (_mercator_y(n2) - _mercator_y(s2)) / lon_span_rad

    # Resolusi cukup: 300 dpi x 2, sisi terpanjang minimal 600 piksel
    target = max(int(max(w_cm, h_cm) / 2.54 * 300 * 2), 600)
    if max(img.size) > target:
        scale = target / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.NEAREST)

    arr = np.array(img)
    transparent = arr[:, :, 3] < 128
    rgb = arr[:, :, :3].copy()
    key = np.array(OVERLAY_KEY_RGB, dtype=rgb.dtype)
    clash = (~transparent) & np.all(rgb == key, axis=2)       # piksel data yang kebetulan sama dengan warna kunci
    rgb[clash] = (key[0] - 2, key[1], key[2])
    rgb[transparent] = key

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    Image.fromarray(rgb, "RGB").save(path, "PNG")
    return path, w2, s2, e2, n2, w_cm, h_cm

def draw_insar_overlays(fig, region, overlays, tmp_files, warnings):
    """Gambar semua GroundOverlay di atas basemap. Return jumlah overlay yang berhasil digambar."""
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
            warnings.append(f"Overlay InSAR #{idx}: gambar tidak dapat diproses ({exc}).")
            continue
        if prep is None:
            warnings.append(f"Overlay InSAR #{idx}: tidak beririsan dengan region peta. Atur region ke area hasil InSAR.")
            continue
        path, w, s, e, n, w_cm, h_cm = prep
        tmp_files.append(path)
        position = f"g{w:.6f}/{s:.6f}+w{w_cm:.4f}c/{h_cm:.4f}c+jBL"
        try:
            try:
                fig.image(imagefile=path, position=position, bitcolor=key_spec)
            except Exception:
                fig.image(imagefile=path, position=position, G=key_spec)   # versi PyGMT tanpa alias 'bitcolor'
            drawn += 1
        except Exception as exc:
            warnings.append(f"Overlay InSAR #{idx} gagal digambar oleh PyGMT: {exc}")
    return drawn

def draw_insar_colorbar(fig, spec, cbar_spec, warnings):
    """Colorbar InSAR native GMT (di bawah peta). Return True jika berhasil."""
    try:
        vmin = float(spec["vmin"])
        vmax = float(spec["vmax"])
    except (KeyError, TypeError, ValueError):
        warnings.append("Colorbar InSAR tidak digambar: nilai Min/Max belum diisi dengan benar.")
        return False
    if not vmax > vmin:
        warnings.append("Colorbar InSAR tidak digambar: nilai Max harus lebih besar dari Min.")
        return False

    cmap = str(spec.get("cmap", "jet"))
    if cmap not in ALLOWED_CMAPS:
        cmap = "jet"
    forbidden = set('+"\'\\@|')   # karakter khusus GMT (modifier '+', kutip, backslash, escape '@', '|')
    label = "".join(ch for ch in str(spec.get("label", "")) if ch not in forbidden)[:80].strip()

    pygmt.makecpt(cmap=cmap, series=[vmin, vmax], reverse=bool(spec.get("reverse", False)), continuous=True)
    with pygmt.config(FORMAT_FLOAT_MAP="%g"):
        try:
            frame = ["a", f"x+l{label}"] if label else ["a"]
            fig.colorbar(position=cbar_spec, box=False, frame=frame)
        except Exception as exc:
            warnings.append(f"Label colorbar InSAR tidak dapat dipakai ({exc}); colorbar digambar tanpa label.")
            fig.colorbar(position=cbar_spec, box=False, frame=["a"])
    return True

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "service": "PyGMT Cartographic API",
        "volcanoes_loaded": len(df_volcanoes),
        "pusgen_segments_loaded": len(pusgen_faults),
        "admin_login_configured": bool(ADMIN_USERNAME and ADMIN_PASSWORD),
        "contours_available": contourpy is not None,
        "insar_overlay_available": Image is not None
    }), 200
@app.route('/api/generate-map', methods=['POST'])
def generate_map():
    tmp_files = []  # file sementara yang dihapus di akhir request
    warnings = []   # catatan untuk pengguna (layer yang diminta tapi tidak bisa digambar)
    try:
        data = request.json
        admin_user = get_admin_from_request()   # None jika bukan administrator
        
        # 1. Parameter Koordinat dengan Normalisasi Longitude
        raw_min_lon = float(data.get('minLon', 121.5))
        raw_max_lon = float(data.get('maxLon', 124.5))
        min_lat = float(data.get('minLat', -9.2))
        max_lat = float(data.get('maxLat', -7.6))

        min_lon = normalize_lon(raw_min_lon)
        max_lon = normalize_lon(raw_max_lon)
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon

        region = [min_lon, max_lon, min_lat, max_lat]
        
        # 2. Parameter Kartografi
        cpt = data.get('cptType', 'topography')
        cbar_pos = data.get('cbarPos', 'bottom')
        frame_style = data.get('frameStyle', 'plain')
        
        # Scalebar
        show_scalebar = data.get('showScalebar', False)
        scale_km = int(data.get('scalebarKm', 50))
        
        # Lapisan Geofisika
        show_faults = data.get('showFaults', False)
        show_volcanoes = data.get('showVolcanoes', False)
        show_contours = data.get('showContours', False)
        show_pusgen = data.get('showPusgen', False)

        # Gempa M>=5 (data dari USGS dimuat di browser lalu dikirim ke sini)
        show_eq = data.get('showEq', False)
        eq_events = data.get('eqEvents', []) or []

        # Overlay InSAR + colorbar (hanya administrator)
        insar_req = data.get('insarOverlay')
        use_insar = False
        if insar_req:
            if admin_user is None:
                warnings.append("Overlay InSAR hanya tersedia untuk administrator; opsi ini diabaikan.")
            elif Image is None:
                warnings.append("Overlay InSAR tidak digambar: paket Pillow belum terpasang di server (tambahkan ke Dockerfile).")
            else:
                use_insar = True

        # GPS Stations (format dari frontend: [{"id": "bako", "lat": -6.309, "lon": 106.203}, ...])
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

        # Inisialisasi PyGMT
        fig = pygmt.Figure()
        proj = "M18c"
        
        # Konfigurasi GMT:
        # FORMAT_GEO_MAP="ddd.xx": maksimal 2 angka di belakang koma, minus untuk arah Selatan
        # FORMAT_FLOAT_MAP="%.0f": angka colorbar BULAT UTUH tanpa koma (integer)
        pygmt.config(
            FORMAT_GEO_MAP="ddd.xx",
            FORMAT_FLOAT_MAP="%.0f"
        )

        # Frame bersih tanpa garis kisi (grid lines)
        frame_code = "f" if frame_style == "zebra" else "p"
        fig_frame = ["a", frame_code]

        # A. Muat Earth Relief (Resolusi adaptif tinggi)
        lon_span = abs(max_lon - min_lon)
        lat_span = abs(max_lat - min_lat)
        if lon_span <= 1.5 and lat_span <= 1.5:
            res = "01s"  # Resolusi detail ~30 meter
        elif lon_span <= 4.5 and lat_span <= 3.5:
            res = "03s"
        else:
            res = "15s"

        grid = pygmt.datasets.load_earth_relief(resolution=res, region=region)
        cmap_name = "geo" if cpt == "terrain" else ("bathy" if cpt == "bathymetry" else "relief")

        is_hillshade = (cpt == "hillshade")
        cpt_path = None

        if is_hillshade:
            # Hillshade abu-abu muda khusus daratan (laut dibiarkan transparan/kosong)
            # Intensitas hillshade (arah cahaya dari barat laut, azimuth 315)
            shade = pygmt.grdgradient(grid=grid, azimuth=315, normalize="t1")
            # Sembunyikan area laut (elevasi < 0 m)
            shade_land = shade.where(grid.values >= 0)

            # CPT gray: intensitas 0 -> abu-abu muda (~194/255), lereng terang -> putih, lereng gelap -> abu tua
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

        # A2. Overlay hasil InSAR (administrator) di atas basemap, di bawah kontur/garis pantai/layer lain
        insar_drawn = 0
        if use_insar:
            insar_drawn = draw_insar_overlays(fig, region, insar_req.get("images", []) or [], tmp_files, warnings)

       # B. Kontur Elevasi: Interval 200m, Anotasi Angka Bulat, Khusus Darat (>= 200m)
        if show_contours:
            fig.grdcontour(
                grid=grid,
                interval=200,
                annotation="200+f7p,Helvetica-Bold,gray20",
                limit=[200, 9000],  # Menggunakan parameter -L untuk rentang elevasi daratan
                pen="0.35p,gray40"
            )

        # Garis Pantai
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

        # D. Sesar Aktif (Bird, 2003)
        if show_faults:
            try:
                fig.plot(
                    data=FAULT_FILE if os.path.exists(FAULT_FILE) else FAULT_URL,
                    pen="1.8p,red3"
                )
            except Exception:
                pass

        # D2. Patahan PusGeN 2024 (hanya segmen yang beririsan dengan region peta)
        if show_pusgen and not pusgen_faults:
            warnings.append("Layer PusGeN tidak digambar: file pusgen_fault_2024.txt tidak ditemukan/terbaca di server.")
        if show_pusgen and pusgen_faults:
            segs_in = [
                sg for sg in pusgen_faults
                if sg["max_lon"] >= min_lon and sg["min_lon"] <= max_lon
                and sg["max_lat"] >= min_lat and sg["min_lat"] <= max_lat
            ]
            if segs_in:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_f:
                    for sg in segs_in:
                        tmp_f.write(">\n")
                        for lon, lat in sg["pts"]:
                            tmp_f.write(f"{lon:.5f} {lat:.5f}\n")
                    pusgen_path = tmp_f.name
                tmp_files.append(pusgen_path)
                fig.plot(data=pusgen_path, pen="1p,magenta3")
            else:
                warnings.append("Tidak ada segmen patahan PusGeN pada region peta ini.")

        # D3. Gempa M>=5 (USGS): lingkaran, ukuran sebanding dengan magnitudo
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
                quakes.sort(key=lambda q: -q[2])   # besar digambar dulu, kecil di atasnya
                fig.plot(
                    x=[q[0] for q in quakes],
                    y=[q[1] for q in quakes],
                    size=[0.08 * max(q[2] - 3.0, 0.5) ** 2 for q in quakes],   # diameter (cm)
                    style="cc",
                    fill="255/165/0@50",
                    pen="0.4p,black"
                )
            else:
                warnings.append("Tidak ada gempa M>=5 di dalam region peta untuk periode yang dipilih.")
        elif show_eq:
            warnings.append("Gempa tidak digambar: data gempa belum dimuat di browser.")

        # E. Gunung Api Holosen (Database GVP)
        if show_volcanoes and df_volcanoes.empty:
            warnings.append("Layer gunung api tidak digambar: database GVP tidak terbaca di server.")
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

        # E2. Stasiun GPS: lingkaran biru (ukuran dalam point)
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
                    # '@' dan '\\' punya arti khusus di teks GMT
                    gps_ids.append(str(st.get('id', '')).replace('\\', '').replace('@', '@@'))
            if gps_lon:
                fig.plot(
                    x=gps_lon,
                    y=gps_lat,
                    style=f"c{gps_size:g}p",
                    fill="blue",
                    pen="0.6p,black"
                )
                # Label ID tepat di atas titik (tepi bawah teks = 2 pt di atas lingkaran)
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
                warnings.append("Tidak ada stasiun GPS yang berada di dalam region peta.")

        # F. Scalebar Southwest (Pojok Kiri Bawah) dengan Transparansi 60%
        if show_scalebar:
            fig.basemap(
                map_scale=f"x0.6c/0.6c+c{min_lat}+w{scale_km}k+lkm+f+jBL",
                box="+gwhite@40+p0.5p,gray30+r2p"
            )

        # G. Colorbar: InSAR (jika overlay aktif) atau elevasi (kecuali mode hillshade)
        cbar_spec = "JMR+w8c/0.5c+o1.0c/0c" if cbar_pos == "right" else "JBC+w10c/0.5c+o0c/1.4c+h"
        if use_insar and insar_drawn > 0:
            draw_insar_colorbar(fig, insar_req, cbar_spec, warnings)
        elif not is_hillshade:
            fig.colorbar(
                position=cbar_spec,
                box=False,
                frame=["a1000", "x+lElevation", "y+lm"]
            )

        # H. Simpan Figure Resolusi 300 DPI
        # PNG transparan (latar belakang di luar peta transparan); JPEG dibuat di frontend dengan latar putih
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            output_png = tmp_png.name
        tmp_files.append(output_png)
        fig.savefig(output_png, dpi=300, transparent=True)

        with open(output_png, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "status": "success",
            "image_base64": f"data:image/png;base64,{encoded_img}",
            "warnings": warnings
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
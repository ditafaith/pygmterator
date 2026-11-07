import os
import io
import tempfile
import base64
import pandas as pd
import numpy as np
import pygmt
from flask import Flask, request, jsonify
from flask_cors import CORS

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

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "service": "PyGMT Cartographic API",
        "volcanoes_loaded": len(df_volcanoes),
        "pusgen_segments_loaded": len(pusgen_faults)
    }), 200
@app.route('/api/generate-map', methods=['POST'])
def generate_map():
    tmp_files = []  # file sementara yang dihapus di akhir request
    warnings = []   # catatan untuk pengguna (layer yang diminta tapi tidak bisa digambar)
    try:
        data = request.json
        
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
        elif lon_span <= 3.5 and lat_span <= 3.5:
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

        # G. Colorbar dengan Angka Bulat Utuh (Tanpa Desimal/Koma)
        if not is_hillshade:
            cbar_spec = "JMR+w8c/0.5c+o1.0c/0c" if cbar_pos == "right" else "JBC+w10c/0.5c+o0c/1.4c+h"
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
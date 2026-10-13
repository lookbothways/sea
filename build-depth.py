#!/usr/bin/env python3
"""Builds depth/<port>.json for each port in ports.json: the height of the bed above chart datum on a
regular grid, for the planner's depth layer. Run by .github/workflows/depth.yml, or by hand:
    pip install rasterio numpy && python build-depth.py [port ...]

Sources, both free:
  - Welsh Government LiDAR terrain model, 1 m, heights above Ordnance Datum Newlyn (DataMapWales,
    Open Government Licence v3). Covers land and the sands that dry; read straight from the
    cloud-optimised GeoTIFF, so only the port's corner of Wales is downloaded.
  - EMODnet Bathymetry digital terrain model, about 115 m, depths below Lowest Astronomical Tide.
    Fills the water that never dries.
Heights are converted to chart datum with each port's cdBelowOdn. LiDAR wins wherever it has a value.

The grid is stored as one byte per cell: z = zMin + v * zStep metres above chart datum, 255 for no data,
rows from north to south, gzipped and base64-encoded. LIDAR_URL and EMODNET_URL override the sources
(a local file path works), for testing."""
import base64, datetime, gzip, json, math, os, sys, urllib.request, warnings
import numpy as np
import rasterio
import rasterio.errors
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

Z_MIN, Z_STEP, NO_DATA = -12.0, 0.1, 255
EMODNET = ('https://ows.emodnet-bathymetry.eu/wcs?service=wcs&version=1.0.0&request=getcoverage&coverage=emodnet:mean'
           '&crs=EPSG:4326&BBOX={w},{s},{e},{n}&format=image/tiff&interpolation=nearest&resx=0.00104166667&resy=0.00104166667')


def log(*a):
    print(*a, flush=True)


def grid_for(port):
    w, s, e, n = port['bbox']
    cell = port.get('cell', 10)
    lat0 = (s + n) / 2
    dlat = cell / 111320
    dlon = cell / (111320 * math.cos(math.radians(lat0)))
    nx, ny = math.ceil((e - w) / dlon), math.ceil((n - s) / dlat)
    return from_origin(w, n, dlon, dlat), (ny, nx), dlon, dlat


def read_lidar(url, bbox, dst_transform, shape, cell):
    """LiDAR heights above ODN, averaged into the grid. NaN where there are none."""
    w, s, e, n = bbox
    out = np.full(shape, np.nan, dtype='float32')
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='3',
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', VSI_CACHE='TRUE')
    with rasterio.Env(**env), rasterio.open(url) as src:
        log(f'LiDAR: {src.width} x {src.height} px at {src.res[0]:g} m, {src.crs}, {src.dtypes[0]}, nodata {src.nodata}, '
            f'overviews {src.overviews(1)}')
        l, b, r, t = transform_bounds('EPSG:4326', src.crs, w, s, e, n, densify_pts=21)
        pad = 4 * cell
        win = from_bounds(l - pad, b - pad, r + pad, t + pad, src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
        # Read at about half the cell size; GDAL picks the matching overview, so little is downloaded.
        f = max(1, int(cell / 2 / abs(src.res[0])))
        rs = (max(1, int(win.height // f)), max(1, int(win.width // f)))
        arr = src.read(1, window=win, out_shape=rs, resampling=Resampling.average, masked=True)
        data = arr.astype('float32').filled(np.nan)
        data[(data < -100) | (data > 3000)] = np.nan          # fill values that are not flagged as nodata
        tr = src.window_transform(win) * Affine.scale(win.width / rs[1], win.height / rs[0])
        log(f'LiDAR window read: {rs[1]} x {rs[0]} px, {np.isfinite(data).mean():.0%} with values')
        reproject(data, out, src_transform=tr, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=dst_transform, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.average)
    return out


def read_emodnet(url, dst_transform, shape):
    """EMODnet elevation relative to LAT (negative below), resampled into the grid."""
    out = np.full(shape, np.nan, dtype='float32')
    if os.path.exists(url):
        raw = open(url, 'rb').read()
    else:
        req = urllib.request.Request(url, headers={'User-Agent': 'sea-depth-builder'})
        raw = urllib.request.urlopen(req, timeout=180).read()
    with MemoryFile(raw) as mf, mf.open() as src:
        arr = src.read(1, masked=True).astype('float32').filled(np.nan)
        arr[(arr < -12000) | (arr > 9000)] = np.nan
        log(f'EMODnet: {src.width} x {src.height} px, {src.crs}, values {np.nanmin(arr):.1f} to {np.nanmax(arr):.1f} m')
        reproject(arr, out, src_transform=src.transform, src_crs=src.crs or 'EPSG:4326', src_nodata=np.nan,
                  dst_transform=dst_transform, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def preview(z, path):
    """A quick look for checking by eye: buff land, greens for sand that dries, blues under chart datum."""
    rgb = np.zeros(z.shape + (3,), dtype='uint8')
    rgb[:] = (200, 200, 200)                                   # no data
    land = z > 6
    dry = (z > 0) & ~land
    wet = z <= 0
    rgb[land] = (226, 214, 180)
    t = np.clip(z / 6, 0, 1)
    for i, (a, b) in enumerate(zip((150, 200, 120), (225, 230, 170))):
        rgb[..., i] = np.where(dry, a + (b - a) * t, rgb[..., i])
    d = np.clip(-z / 15, 0, 1)
    for i, (a, b) in enumerate(zip((190, 225, 245), (40, 90, 160))):
        rgb[..., i] = np.where(wet, a + (b - a) * d, rgb[..., i])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path, 'w', driver='PNG', width=z.shape[1], height=z.shape[0], count=3, dtype='uint8') as dst:
            dst.write(np.moveaxis(rgb, 2, 0))


def build(pid, port):
    log(f'--- {pid}: {port.get("name", pid)}')
    tr, shape, dlon, dlat = grid_for(port)
    w, s, e, n = port['bbox']
    lidar = read_lidar(os.environ.get('LIDAR_URL', port['lidar']), port['bbox'], tr, shape, port.get('cell', 10))
    cd_lidar = lidar + port['cdBelowOdn']
    pad = 0.01
    emod = read_emodnet(os.environ.get('EMODNET_URL', EMODNET.format(w=w - pad, s=s - pad, e=e + pad, n=n + pad)), tr, shape)
    # The DTM should be negative under water. If it comes the other way up, turn it over.
    sea = ~np.isfinite(lidar) & np.isfinite(emod)
    if sea.any() and np.nanmedian(emod[sea]) > 0:
        log('EMODnet values look positive-down, flipping the sign')
        emod = -emod
    z = np.where(np.isfinite(cd_lidar), cd_lidar, emod)
    v = np.where(np.isfinite(z), np.clip(np.round((z - Z_MIN) / Z_STEP), 0, 254), NO_DATA).astype('uint8')
    fin = np.isfinite(z)
    stats = {
        'cells': int(z.size), 'lidar': round(float(np.isfinite(lidar).mean()), 3),
        'emodnet': round(float((~np.isfinite(lidar) & np.isfinite(emod)).mean()), 3), 'none': round(float((~fin).mean()), 3),
        'zRange': [round(float(np.nanmin(z)), 2), round(float(np.nanmax(z)), 2)] if fin.any() else None,
    }
    log('cells from LiDAR {lidar:.1%}, from EMODnet {emodnet:.1%}, none {none:.1%}'.format(**stats))
    out = {
        'port': pid, 'name': port.get('name', pid), 'bbox': [w, n - shape[0] * dlat, w + shape[1] * dlon, n],
        'nx': shape[1], 'ny': shape[0], 'cell': port.get('cell', 10), 'rows': 'north-first',
        'zMin': Z_MIN, 'zStep': Z_STEP, 'noData': NO_DATA,
        'datum': f'chart datum, taken as {port["cdBelowOdn"]} m below Ordnance Datum Newlyn',
        'sources': ['Welsh Government LiDAR terrain model (DataMapWales), Open Government Licence v3',
                    'EMODnet Bathymetry Consortium, EMODnet Digital Bathymetry (DTM)'],
        'built': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'stats': stats,
        'z': base64.b64encode(gzip.compress(v.tobytes(), 9, mtime=0)).decode('ascii'),
    }
    os.makedirs('depth', exist_ok=True)
    with open(f'depth/{pid}.json', 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    preview(z, f'depth/{pid}-preview.png')
    log(f'wrote depth/{pid}.json, {os.path.getsize(f"depth/{pid}.json") / 1024:.0f} KB, grid {shape[1]} x {shape[0]}')


if __name__ == '__main__':
    ports = json.load(open('ports.json'))
    for pid in (sys.argv[1:] or list(ports)):
        build(pid, ports[pid])

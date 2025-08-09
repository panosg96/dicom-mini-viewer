from io import BytesIO
import numpy as np
import pydicom as pdcm
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage import exposure
from PIL import Image
import os


def read_dicom(path):
    ds = pdcm.dcmread(path, force=True)
    return ds


def dataset_to_numpy(ds, voi_lut=True, fix_monochrome=True):
    try:
        img = ds.pixel_array
    except Exception:
        raise

    if voi_lut:
        try:
            img = apply_voi_lut(img, ds)
        except Exception:
            pass

    # rescale
    intercept_element = ds.get(('0028', '1052'))
    if intercept_element is not None:
        intercept = float(intercept_element.value)
    else:
        intercept = float(ds.get('RescaleIntercept', 0))

    slope_element = ds.get(('0028', '1053'))
    if slope_element is not None:
        slope = float(slope_element.value)
    else:
        slope = float(ds.get('RescaleSlope', 1))

    img = img*slope + intercept 

    # handle photometric interpretation
    photometric = ds.get('PhotometricInterpretation', '').upper()
    if fix_monochrome and photometric == 'MONOCHROME1':
        img = np.max(img) - img

    # normalize 0-255
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    img = (img*255).astype(np.uint8)

    # compute basic window center/width if present 
    wc = None
    ww = None
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        # WindowCenter/Width may be sequences 
        if isinstance(wc, pdcm.multival.MultiValue) or isinstance(wc, (list, tuple)):
            wc = float(wc[0])
        if isinstance(ww, pdcm.multival.MultiValue) or isinstance(ww, (list, tuple)):
            ww = float(ww[0])
    
    return img, wc, ww

def apply_window(img, center=None, width=None):
    
    if center is None or width is None:
        return img
    
    img = img.astype(np.int32)
    lo = center - width / 2.0
    hi = center + width / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo)/(hi - lo)
    img = (img * 255).astype(np.uint8)
    return img 

def load_series_from_folder(folder_path):
    dicom_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith('.dcm') or f.lower().endswith('.dicom') or '.' not in f:
                dicom_files.append(os.path.join(root, f))


    # quick filter: try reading headers only
    series = {}
    for fp in dicom_files:
        try:
            ds = pdcm.dcmread(fp, stop_before_pixels=True, force=True)
            series_uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
            series.setdefault(series_uid, []).append(fp)
        except Exception:
            continue

    for uid, paths in series.items():
        try:
            paths_sorted = sorted(paths, key=lambda p: int(getattr(pdcm.dcmread(p, stop_before_pixels=True, force=True), 'InstanceNumber', 0)))
            series[uid] = paths_sorted
        except Exception:
            series[uid] = paths

    return series

def export_selected_slices_csv(rows, out_path):

    import csv
    if not rows:
        return
    keys = rows[0].keys()
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


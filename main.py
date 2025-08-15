import streamlit as st
from dicom_utils import read_dicom, dataset_to_numpy, apply_window, load_series_from_folder, export_selected_slices_csv
import os
from zipfile import ZipFile
import tempfile
import numpy as np
from PIL import Image
import io
import pandas as pd


st.set_page_config(layout='wide', page_title='PACS Mini Viewer')

st.header('PACS Mini Viewer', divider="blue")

st.markdown('Lightweight DICOM viewer with simple window/level controls. Local-only demo')

# Sidebar: upload
st.sidebar.header('Data')
uploaded_files = st.sidebar.file_uploader('Upload DICOM files (multiple) or a ZIP of DICOMs', accept_multiple_files=True, type=['dcm','dicom','zip'])

# Use a temp dir to save uploaded files for processing
if 'tmpdir' not in st.session_state:
    st.session_state['tmpdir'] = tempfile.TemporaryDirectory()

tmpdir = st.session_state['tmpdir'].name


filepaths = []
if uploaded_files:
    for uf in uploaded_files:
        filename = uf.name
        fp = os.path.join(tmpdir, filename)
        with open(fp, 'wb') as f:
            f.write(uf.getbuffer())
        # if zip, extract
        if filename.lower().endswith('.zip'):
            try:
                with ZipFile(fp, 'r') as zf:
                    zf.extractall(tmpdir)
            except Exception as e:
                st.sidebar.error(f'Error extracting zip: {e}')


# Option to load from local folder path (for dev)
local_folder = st.sidebar.text_input('Or provide local folder path (dev)', '')
if local_folder:
    filepaths.append(local_folder)

# If any files/dirs in tmpdir, scan
series_dict = {}
if os.listdir(tmpdir):
    series_dict = load_series_from_folder(tmpdir)

# Also check if user provided local folder path
if local_folder and os.path.isdir(local_folder):
    series_dict = load_series_from_folder(local_folder)

if not series_dict:
    st.info('Upload DICOM files or a zip containing DICOMs.')
    st.stop()

# Let user choose series
series_uids = list(series_dict.keys())
selected_series = st.selectbox('Select Series (SeriesInstanceUID)', series_uids)
files_in_series = series_dict[selected_series]

# quick metadata table
st.sidebar.write(f'Files in series: {len(files_in_series)}')

# Build thumbnails for quick navigation
thumbs = []
for f in files_in_series[:50]:
    try:
        ds = read_dicom(f)
        img, wc, ww = dataset_to_numpy(ds)
        thumbs.append({'path': f, 'shape': img.shape})
    except Exception:
        thumbs.append({'path': f, 'shape': None})

# main layout
cont1 = st.container()
cont2 = st.container()
cont3 = st.container(horizontal=True, horizontal_alignment='center')
cont4 = st.container(horizontal=True, horizontal_alignment='center')

with cont1:
    st.write('### Series Slices')

    num_slices = len(files_in_series)

    if num_slices == 0:
        st.error("No image slices found in this series.")
        st.stop()

    elif num_slices == 1:
        st.write("Single slice available.")
        idx = 0
        file_to_show = files_in_series[0]

    else:
        idx = st.slider(
            'Slice',
            min_value=0,
            max_value=num_slices - 1,
            value=0
        )
        file_to_show = files_in_series[int(idx)]


    

with cont2:
    st.write('### Image Viewer')
    ds = read_dicom(file_to_show)
    img, wc, ww = dataset_to_numpy(ds)

    # window/level sliders
    if wc is None:
        wc = 128
    if ww is None:
        ww = 256
    wc = st.slider('Window Center', min_value=0, max_value=1024, value=200)
    ww = st.slider('Window Width', min_value=1, max_value=2048, value=400)
  
    
with cont3:    
    out_img = apply_window(img, center=wc, width=ww)
    pil_img = Image.fromarray(out_img)
    st.image(pil_img, caption=f'Slice {idx}', use_container_width=False, width=250)


with cont4:

    # Optionally list the DICOM tags of current slice
    if st.checkbox('Show DICOM tags (current slice)'):
        try:
            ds = read_dicom(file_to_show)
            tags = []
            for elem in ds.iterall():
                if elem.VR != 'OB' and elem.VR != 'OW' and elem.tag.group != 0x7fe0:
                    tags.append({'name': elem.name, 'tag': str(elem.tag), 'value': str(elem.value)[:200]})
            df = pd.DataFrame(tags)
            st.dataframe(df)
        except Exception as e:
            st.error(f'Error reading tags: {e}')

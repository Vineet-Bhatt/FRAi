import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import struct

def parse_csv(path):
    # assume two columns frequency,magnitude or headered
    df = pd.read_csv(path)
    if 'frequency' in df.columns.str.lower():
        freq_col = [c for c in df.columns if c.lower().startswith('freq')][0]
        mag_col = [c for c in df.columns if 'mag' in c.lower() or 'response' in c.lower() or 'amp' in c.lower()][0]
        freqs = df[freq_col].values
        mags = df[mag_col].values
    else:
        # fallback: first two columns
        freqs = df.iloc[:,0].values
        mags = df.iloc[:,1].values
    return freqs.astype(float), mags.astype(float)

def parse_xml(path):
    # vendor XML shape varies; here's an example where <point freq="..." mag="..."/>
    tree = ET.parse(path)
    root = tree.getroot()
    freqs, mags = [], []
    for p in root.findall('.//point'):
        freqs.append(float(p.get('freq')))
        mags.append(float(p.get('mag')))
    return np.array(freqs), np.array(mags)

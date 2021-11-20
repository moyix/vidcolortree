#!/usr/bin/env python

import cv2
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import squarify
import subprocess
import argparse
from sklearn.cluster import KMeans, MiniBatchKMeans
import av

# Pixel ordering for different backends
colorformats = {
    'pyav': [0,1,2],   # RGB
    'opencv': [2,1,0], # BGR
}

def hexcolor(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

# Argument parser with one required positional argument
parser = argparse.ArgumentParser(description='Video Histogram')
parser.add_argument('vid_filename', help='Video file to process')
# Number of palette colors to use
parser.add_argument('-c', '--colors', type=int, default=256, help='Number of colors to use')
# Option to force regeneration of histogram
parser.add_argument('-f', '--force', action='store_true', help='Force recomputation of the histogram')
# Only look at I-frames
parser.add_argument('-i', '--iframes', action='store_true', help='Only look at I-frames')
# Which backend to use (opencv or pyav)
parser.add_argument('-b', '--backend', choices=['opencv', 'pyav'], default='opencv',
    help='Which backend to use (PyAV or OpenCV)')
# Which clustering algorithm to use
parser.add_argument('-m', '--method', choices=['kmeans', 'mbkmeans'], default='mbkmeans',
    help='Which clustering algorithm to use (K-Means or MiniBatchKMeans)')
# Where to save output files
parser.add_argument('-d', '--output_dir', default=None, help='Output directory')
# Ignore colors that are too close to black or white
parser.add_argument('-t', '--threshold', type=int, default=0, help='Ignore colors that are within this distance of black/white (Euclidean)')
# Ignore colors that are more than a particular percentage of the total
parser.add_argument('-p', '--percent', type=float, default=100.0, help='Ignore colors that are more than this percent of the video')

args = parser.parse_args()
ClusterAlg = {
    'kmeans': KMeans,
    'mbkmeans': MiniBatchKMeans
}[args.method]

if args.percent > 100.0 or args.percent < 0.0:
    parser.error('Percentage must be between 0 and 100')

vid_filename = Path(args.vid_filename)
if not vid_filename.is_file():
    print("Error: File not found:", vid_filename, file=sys.stderr)
    sys.exit(1)

def iter_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def iter_iframes(cap):
    frame_types = get_frame_types(str(vid_filename))
    i_frames = [i for i, t in frame_types if t == 'I']
    for i in i_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
        yield frame

BATCH_SIZE = 1000

def get_hist_opencv(filename, iframes=False):
    hist = np.zeros((256,256,256))
    framefn = iter_iframes if args.iframes else iter_frames
    i = 0
    batch = []
    for frame in framefn(cap):
        batch.append(frame)
        i += 1
        if i == BATCH_SIZE:
            hist += cv2.calcHist(batch, [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
            batch = []
            i = 0
    if batch: hist += cv2.calcHist(batch, [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
    cap.release()
    return hist

def get_hist_pyav(filename, iframes=False):
    hist = np.zeros((256,256,256))
    with av.open(filename) as container:
        stream = container.streams.video[0]
        if iframes: stream.codec_context.skip_frame = 'NONREF'
        stream.thread_type = 'AUTO'
        i = 0
        batch = []
        for frame in container.decode(stream):
            batch.append(frame.to_ndarray(format='rgb24'))
            i += 1
            if i == BATCH_SIZE:
                hist += cv2.calcHist(batch, [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
                batch = []
                i = 0
        if batch: hist += cv2.calcHist(batch, [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
    return hist

if args.output_dir:
    hist_filename = (Path(args.output_dir) / vid_filename.name).with_suffix('.hist.npz')
else:
    hist_filename = vid_filename.with_suffix('.hist.npz')

if args.force or not hist_filename.exists():
    cap = cv2.VideoCapture(str(vid_filename))
    print("Calculating video histogram...")
    if args.backend == 'opencv':
        hist = get_hist_opencv(str(vid_filename),iframes=args.iframes)
    elif args.backend == 'pyav':
        hist = get_hist_pyav(str(vid_filename),iframes=args.iframes)
    else:
        print("Error: Invalid backend:", args.backend, file=sys.stderr)
        sys.exit(1)
    np.savez_compressed(str(hist_filename), hist=hist)
else:
    hist = np.load(str(hist_filename))['hist']

# Initial list of colors in the video
all_colors = np.argwhere(hist).astype(np.uint8)

# Filter out colors that are too close to black or white
if args.threshold:
    print("Filtering out colors that are too close to black...", end='', flush=True)
    all_colors = all_colors[np.sqrt(np.sum(np.square(np.abs(all_colors - [0,0,0])), axis=1)) > args.threshold]
    print(f"{len(all_colors)} colors remain")
    print("Filtering out colors that are too close to white...", end='', flush=True)
    all_colors = all_colors[np.sqrt(np.sum(np.square(np.abs(all_colors - [255,255,255])), axis=1)) > args.threshold]
    print(f"{len(all_colors)} colors remain")

# Use K-Means to cluster the colors
print(f"Clustering to select {args.colors} dominant colors...")
kmeans = ClusterAlg(n_clusters=args.colors, random_state=0)
kmeans.fit(all_colors, sample_weight=hist[all_colors[:,0],all_colors[:,1],all_colors[:,2]])
palette = np.rint(kmeans.cluster_centers_).astype(np.uint8)
# Convert backend's color format to RGB
palette = palette[:,colorformats[args.backend]]
palhist = np.zeros(args.colors)
palidx = kmeans.predict(all_colors)
for color, idx in zip(all_colors, palidx):
    palhist[idx] += hist[color[0], color[1], color[2]]
palhist /= palhist.sum()

# Filter out colors that are too dominant (more than a certain percentage of the video after clustering)
if args.percent < 100.0:
    print("Filtering out colors that are too dominant...", end='', flush=True)
    before = len(palhist[palhist > 0.0])
    palhist[palhist*100.0 > args.percent] = 0.0
    after = len(palhist[palhist > 0.0])
    print(f"palette reduced from {before} to {after} colors")

palette = palette.astype(np.float32) / 255
# Use squarify to plot the histogram using the palette colors.
# Skip any colors with zero frequency.
squarify.plot(sizes=palhist[palhist > 0], color=palette[palhist > 0])
plt.axis('off')
if args.output_dir:
    vidhist_filename = (Path(args.output_dir) / vid_filename.name).with_suffix('.hist.png')
else:
    vidhist_filename = vid_filename.with_suffix('.hist.png')
plt.savefig(str(vidhist_filename), bbox_inches='tight', pad_inches=0)
print(f"Saved video colors to {vidhist_filename}")
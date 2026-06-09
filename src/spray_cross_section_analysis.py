"""
Spray-Occupied Area, Width & Height Analysis
Mie Scattering Cut-Section Images - LJICF Spray Bar

Method:
  1. Compute mean image of all raw Mie scattering frames
  2. Estimate background from spray-free edge regions of the mean image
     (simulates "laser on, no spray" condition)
  3. Subtract background from the mean image
  4. Interpolate across the spray bar band to fix over-subtraction
  5. Otsu threshold on the mean-subtracted image for time-averaged spray boundary
  6. Per-frame processing for instantaneous statistics and probability map
  7. All results referenced to nozzle exit coordinate system
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import csv
import sys
from scipy.ndimage import gaussian_filter, uniform_filter1d

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# ====================== USER SETTINGS ======================
INPUT_DIR = Path('')
OUTPUT_DIR = Path("")
PIXEL_TO_MM = 0.05995835  # mm per pixel
X_ORIGIN_FROM_LEFT_MM = 39.932  # X distance of nozzle exit from left edge of image (mm)
Y_ORIGIN_FROM_TOP_MM = 52.044  # Y distance of nozzle exit from top edge of image (mm)
SAVE_FORMAT = "png"    # "png" or "pdf" for vector
DPI = 300
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Publication plot style (journal-quality: ASME / Elsevier standard) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.5,
    'mathtext.default': 'regular',
})

# --- Fixed axis ranges for cross-case comparison ---
X_LIM = (-42, 42)    # mm, centered on nozzle exit
Y_LIM = (-30, 55)    # mm, nozzle at origin (positive upward)

# ===================================================================
# STEP 1: Load all images and compute mean
# ===================================================================
image_files = sorted(INPUT_DIR.glob("*.JPG"))
n_frames = len(image_files)
print(f"Found {n_frames} images")

sample = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
h, w = sample.shape
print(f"Image size: {w} x {h} pixels ({w*PIXEL_TO_MM:.1f} x {h*PIXEL_TO_MM:.1f} mm)")

all_images = []
for f in image_files:
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float64)
    all_images.append(img)
all_images = np.array(all_images)  # shape: (N, H, W)

mean_image = np.mean(all_images, axis=0)
print(f"Mean image computed from {n_frames} frames")

# ===================================================================
# STEP 1b: Nozzle exit origin (fixed coordinates)
# ===================================================================
image_height_mm = h * PIXEL_TO_MM
image_width_mm = w * PIXEL_TO_MM
x_origin_px = X_ORIGIN_FROM_LEFT_MM / PIXEL_TO_MM
nozzle_row_px = Y_ORIGIN_FROM_TOP_MM / PIXEL_TO_MM

print(f"\nImage size: {image_width_mm:.1f} x {image_height_mm:.1f} mm ({w} x {h} px)")
print(f"Nozzle exit X: {X_ORIGIN_FROM_LEFT_MM} mm from left = pixel {x_origin_px:.1f}")
print(f"Nozzle exit Y: {Y_ORIGIN_FROM_TOP_MM} mm from top = row {nozzle_row_px:.1f} from top")

# Save origin info
origin_file = OUTPUT_DIR / "origin_coordinates.txt"
with open(origin_file, 'w') as f:
    f.write(f"Nozzle exit origin (fixed):\n")
    f.write(f"  x_origin_px = {x_origin_px:.1f} (from left)\n")
    f.write(f"  nozzle_row_px = {nozzle_row_px:.1f} (from top)\n")
    f.write(f"  X_from_left = {X_ORIGIN_FROM_LEFT_MM:.3f} mm\n")
    f.write(f"  Y_from_top = {Y_ORIGIN_FROM_TOP_MM:.3f} mm\n")
    f.write(f"  Pixel_to_mm = {PIXEL_TO_MM}\n")
    f.write(f"  Image size = {image_width_mm:.1f} x {image_height_mm:.1f} mm\n")

# ===================================================================
# Coordinate transform functions
# ===================================================================
# Image pixel (col, row) -> physical (x_mm, y_mm) with nozzle at origin
# Y increases UPWARD from nozzle (positive = above nozzle)
# X: positive to the right of nozzle

def px_to_mm_x(col_px):
    """Pixel column -> x in mm (nozzle = 0)"""
    return (col_px - x_origin_px) * PIXEL_TO_MM

def px_to_mm_y(row_px):
    """Pixel row -> y in mm (nozzle = 0, positive upward)
    Both nozzle_row_px and row_px are measured from the top of the image,
    so (nozzle_row_px - row_px) is positive when row_px is above the nozzle.
    """
    return (nozzle_row_px - row_px) * PIXEL_TO_MM

# Extent for imshow: [x_left, x_right, y_bottom, y_top]
# With origin='upper': first row (row 0) at y_top, last row (row h) at y_bottom
x_left_mm = px_to_mm_x(0)
x_right_mm = px_to_mm_x(w)
y_top_mm = px_to_mm_y(0)         # top of image (row 0)
y_bottom_mm = px_to_mm_y(h)      # bottom of image (row h)
extent_mm = [x_left_mm, x_right_mm, y_bottom_mm, y_top_mm]

print(f"\nCoordinate system (nozzle exit = origin):")
print(f"  X range: [{x_left_mm:.1f}, {x_right_mm:.1f}] mm")
print(f"  Y range: [{y_bottom_mm:.1f}, {y_top_mm:.1f}] mm")

# ===================================================================
# STEP 2: Estimate background from spray-free edges of the mean image
# ===================================================================
edge_margin = 80  # pixels from each side (spray-free)

row_bg_left = np.mean(mean_image[:, :edge_margin], axis=1)
row_bg_right = np.mean(mean_image[:, -edge_margin:], axis=1)

# Smooth vertically (preserve bar peak)
row_bg_avg = (row_bg_left + row_bg_right) / 2.0
row_bg_trend = uniform_filter1d(row_bg_avg, size=100)
row_residual = row_bg_avg - row_bg_trend
spray_bar_center = np.argmax(row_residual)
peak_residual = row_residual[spray_bar_center]

bar_candidate = np.where(row_residual > 0.20 * peak_residual)[0]
diffs = np.diff(bar_candidate)
clusters = np.split(bar_candidate, np.where(diffs > 10)[0] + 1)
for cl in clusters:
    if spray_bar_center in cl:
        bar_rows_cluster = cl
        break

bar_top = max(0, bar_rows_cluster[0] - 5)
bar_bot = min(h - 1, bar_rows_cluster[-1] + 5)

# Smooth left and right profiles (preserve bar)
bar_vals_left = row_bg_left[bar_top:bar_bot+1].copy()
bar_vals_right = row_bg_right[bar_top:bar_bot+1].copy()

row_bg_left_smooth = uniform_filter1d(row_bg_left, size=30)
row_bg_right_smooth = uniform_filter1d(row_bg_right, size=30)

row_bg_left_smooth[bar_top:bar_bot+1] = bar_vals_left
row_bg_right_smooth[bar_top:bar_bot+1] = bar_vals_right

row_bg_left_final = gaussian_filter(row_bg_left_smooth, sigma=3)
row_bg_right_final = gaussian_filter(row_bg_right_smooth, sigma=3)

margin_bar = 3
if len(bar_vals_left) > 2 * margin_bar:
    cs = bar_top + margin_bar
    cl = len(bar_vals_left[margin_bar:-margin_bar])
    row_bg_left_final[cs:cs+cl] = bar_vals_left[margin_bar:-margin_bar]
    row_bg_right_final[cs:cs+cl] = bar_vals_right[margin_bar:-margin_bar]

# Interpolate across full width
x_left_anchor = edge_margin / 2.0
x_right_anchor = w - edge_margin / 2.0
x_all = np.arange(w, dtype=np.float64)

background = np.zeros((h, w), dtype=np.float64)
for row in range(h):
    val_left = row_bg_left_final[row]
    val_right = row_bg_right_final[row]
    background[row, :] = val_left + (val_right - val_left) * \
                          (x_all - x_left_anchor) / (x_right_anchor - x_left_anchor)
    background[row, :int(x_left_anchor)] = val_left
    background[row, int(x_right_anchor):] = val_right

dark_level = np.median(background[:bar_top, :])
bar_intensity = np.mean(background[bar_top:bar_bot+1, :])
print(f"\nBackground: edge sampling + linear interpolation")
print(f"  Sensor dark level: {dark_level:.1f}")
print(f"  Spray bar: rows {bar_top}-{bar_bot} "
      f"(y = {px_to_mm_y(bar_top):.1f} to {px_to_mm_y(bar_bot):.1f} mm)")
print(f"  Bar reflection intensity: {bar_intensity:.1f}")

# ===================================================================
# STEP 3: Subtract background from the MEAN image
# ===================================================================
mean_sub_float = np.clip(mean_image - background, 0, 255)

# Fix bar-region black strip via vertical interpolation
interp_margin = 5
row_above = max(0, bar_top - interp_margin)
row_below = min(h - 1, bar_bot + interp_margin)

above_vals = np.mean(mean_sub_float[row_above:bar_top, :], axis=0)
below_vals = np.mean(mean_sub_float[bar_bot+1:row_below+1, :], axis=0)

for row in range(bar_top, bar_bot + 1):
    t = (row - bar_top) / max(1, bar_bot - bar_top)
    mean_sub_float[row, :] = above_vals * (1 - t) + below_vals * t

print(f"Bar region interpolated vertically")
mean_subtracted = mean_sub_float.astype(np.uint8)


# STEP 4: Otsu threshold on the mean-subtracted image

mean_sub_blurred = cv2.GaussianBlur(mean_subtracted, (5, 5), 0)
otsu_threshold, _ = cv2.threshold(mean_sub_blurred, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold: {otsu_threshold}")

_, mean_binary = cv2.threshold(mean_sub_blurred, int(otsu_threshold), 255,
                                cv2.THRESH_BINARY)

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
min_blob_area = 100

mean_binary = cv2.morphologyEx(mean_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
mean_binary = cv2.morphologyEx(mean_binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mean_binary, connectivity=8)
mean_binary_clean = np.zeros_like(mean_binary)
for lbl in range(1, num_labels):
    if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_area:
        mean_binary_clean[labels == lbl] = 255
mean_binary = mean_binary_clean


# STEP 5: Time-averaged spray metrics from the mean binary mask

spray_pixels = np.where(mean_binary > 0)
if len(spray_pixels[0]) > 0:
    mean_area_px = len(spray_pixels[0])
    mean_top_row = spray_pixels[0].min()
    mean_bot_row = spray_pixels[0].max()
    mean_height_px = mean_bot_row - mean_top_row + 1
    mean_left_col = spray_pixels[1].min()
    mean_right_col = spray_pixels[1].max()
    mean_width_px = mean_right_col - mean_left_col + 1
else:
    mean_area_px = mean_height_px = mean_width_px = 0
    mean_top_row = mean_bot_row = mean_left_col = mean_right_col = 0

mean_area_mm2 = mean_area_px * PIXEL_TO_MM**2
mean_width_mm = mean_width_px * PIXEL_TO_MM
mean_height_mm = mean_height_px * PIXEL_TO_MM

# Width profile (row by row)
width_profile_px = np.zeros(h)
left_boundary = np.full(h, np.nan)
right_boundary = np.full(h, np.nan)
for row in range(h):
    cols = np.where(mean_binary[row, :] > 0)[0]
    if len(cols) > 0:
        left_boundary[row] = cols[0]
        right_boundary[row] = cols[-1]
        width_profile_px[row] = cols[-1] - cols[0] + 1

width_profile_mm = width_profile_px * PIXEL_TO_MM
y_phys = np.array([px_to_mm_y(row) for row in range(h)])  # physical Y for each row

max_w_row = np.argmax(width_profile_px)
max_width_mm = width_profile_mm[max_w_row]

# Spray extents in physical coordinates
spray_y_top_mm = px_to_mm_y(mean_top_row)    # highest point (positive)
spray_y_bot_mm = px_to_mm_y(mean_bot_row)    # lowest point
spray_height_phys = spray_y_top_mm - spray_y_bot_mm

print(f"\n{'='*60}")
print(f"  TIME-AVERAGED SPRAY METRICS (nozzle exit = origin)")
print(f"{'='*60}")
print(f"  Spray area:       {mean_area_px} px  =  {mean_area_mm2:.1f} mm2")
print(f"  Max width:        {width_profile_px[max_w_row]:.0f} px  =  {max_width_mm:.1f} mm")
print(f"  Height:           {mean_height_px} px  =  {spray_height_phys:.1f} mm")
print(f"  Spray top:        y = {spray_y_top_mm:.1f} mm")
print(f"  Spray bottom:     y = {spray_y_bot_mm:.1f} mm")
print(f"  Otsu threshold:   {otsu_threshold}")
print(f"{'='*60}")


# STEP 6: Per-frame processing

print("\nProcessing individual frames...")
binary_masks = np.zeros((n_frames, h, w), dtype=np.uint8)
areas_px = []
widths_px = []
heights_px = []

for i in range(n_frames):
    frame_sub_f = np.clip(all_images[i] - background, 0, 255)
    above_f = np.mean(frame_sub_f[row_above:bar_top, :], axis=0)
    below_f = np.mean(frame_sub_f[bar_bot+1:row_below+1, :], axis=0)
    for row in range(bar_top, bar_bot + 1):
        t = (row - bar_top) / max(1, bar_bot - bar_top)
        frame_sub_f[row, :] = above_f * (1 - t) + below_f * t
    frame_sub = frame_sub_f.astype(np.uint8)
    blurred = cv2.GaussianBlur(frame_sub, (5, 5), 0)
    _, binary = cv2.threshold(blurred, int(otsu_threshold), 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    num_lab, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lbl in range(1, num_lab):
        if st[lbl, cv2.CC_STAT_AREA] >= min_blob_area:
            cleaned[lab == lbl] = 255
    binary_masks[i] = cleaned

    sp = np.where(cleaned > 0)
    if len(sp[0]) > 0:
        area = len(sp[0])
        height = sp[0].max() - sp[0].min() + 1
        width = sp[1].max() - sp[1].min() + 1
    else:
        area = width = height = 0
    areas_px.append(area)
    widths_px.append(width)
    heights_px.append(height)

areas_px = np.array(areas_px)
widths_px = np.array(widths_px)
heights_px = np.array(heights_px)
areas_mm2 = areas_px * PIXEL_TO_MM**2
widths_mm = widths_px * PIXEL_TO_MM
heights_mm = heights_px * PIXEL_TO_MM

# Spray probability map
spray_probability = np.mean(binary_masks.astype(np.float64) / 255.0, axis=0)

print(f"  Per-frame area:   {np.mean(areas_mm2):.1f} +/- {np.std(areas_mm2):.1f} mm2")
print(f"  Per-frame width:  {np.mean(widths_mm):.1f} +/- {np.std(widths_mm):.1f} mm")
print(f"  Per-frame height: {np.mean(heights_mm):.1f} +/- {np.std(heights_mm):.1f} mm")


# Common display settings

vmin_display = 0
vmax_display = float(np.max(mean_image))

# Axis labels (nozzle-referenced)
xlabel_str = '$x$ (mm)'
ylabel_str = '$y$ (mm)'


# Helper: convert contour pixels to physical mm coordinates

def contour_to_mm(cnt):
    """Convert OpenCV contour (pixel) to physical mm coordinates."""
    cnt_mm = cnt.astype(np.float64).copy()
    cnt_mm[:, 0, 0] = px_to_mm_x(cnt[:, 0, 0].astype(np.float64))
    cnt_mm[:, 0, 1] = px_to_mm_y(cnt[:, 0, 1].astype(np.float64))
    return cnt_mm


# FIGURE 1: Mean Raw Image

fig, ax = plt.subplots(figsize=(4.5, 7))
ax.imshow(mean_image.astype(np.uint8), cmap='gray', extent=extent_mm, aspect='equal')
ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
ax.set_title('Mean Mie Scattering Image')
ax.set_xlim(X_LIM); ax.set_ylim(Y_LIM)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_mean_image.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_mean_image.{SAVE_FORMAT}")


# FIGURE 2: Estimated Background

fig, ax = plt.subplots(figsize=(4.5, 7))
ax.imshow(background.astype(np.uint8), cmap='gray', extent=extent_mm, aspect='equal',
          vmin=vmin_display, vmax=vmax_display)
ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
ax.set_title('Estimated Background')
ax.set_xlim(X_LIM); ax.set_ylim(Y_LIM)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_background_image.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_background_image.{SAVE_FORMAT}")


# FIGURE 3: Mean - Background

fig, ax = plt.subplots(figsize=(4.5, 7))
ax.imshow(mean_subtracted, cmap='gray', extent=extent_mm, aspect='equal')
ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
ax.set_title('Mean Image $-$ Background')
ax.set_xlim(X_LIM); ax.set_ylim(Y_LIM)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_mean_subtracted.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_mean_subtracted.{SAVE_FORMAT}")


# FIGURE 4: Processing Pipeline (4-panel)

fig, axes = plt.subplots(2, 2, figsize=(9, 12))

axes[0, 0].imshow(mean_image.astype(np.uint8), cmap='gray', extent=extent_mm, aspect='equal',
                  vmin=vmin_display, vmax=vmax_display)
axes[0, 0].set_title('(a) Mean Raw Image')
axes[0, 0].set_xlabel(xlabel_str); axes[0, 0].set_ylabel(ylabel_str)
axes[0, 0].set_xlim(X_LIM); axes[0, 0].set_ylim(Y_LIM)

axes[0, 1].imshow(background.astype(np.uint8), cmap='gray', extent=extent_mm, aspect='equal',
                  vmin=vmin_display, vmax=vmax_display)
axes[0, 1].set_title('(b) Estimated Background')
axes[0, 1].set_xlabel(xlabel_str); axes[0, 1].set_ylabel(ylabel_str)
axes[0, 1].set_xlim(X_LIM); axes[0, 1].set_ylim(Y_LIM)

axes[1, 0].imshow(mean_subtracted, cmap='gray', extent=extent_mm, aspect='equal',
                  vmin=vmin_display, vmax=vmax_display)
axes[1, 0].set_title('(c) Mean $-$ Background')
axes[1, 0].set_xlabel(xlabel_str); axes[1, 0].set_ylabel(ylabel_str)
axes[1, 0].set_xlim(X_LIM); axes[1, 0].set_ylim(Y_LIM)

axes[1, 1].imshow(mean_binary, cmap='gray', extent=extent_mm, aspect='equal')
axes[1, 1].set_title(f'(d) Binary Mask (Otsu = {otsu_threshold:.0f})')
axes[1, 1].set_xlabel(xlabel_str); axes[1, 1].set_ylabel(ylabel_str)
axes[1, 1].set_xlim(X_LIM); axes[1, 1].set_ylim(Y_LIM)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_processing_pipeline.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_processing_pipeline.{SAVE_FORMAT}")


# FIGURE 5: Spray Probability Map

fig, ax = plt.subplots(figsize=(5, 7))
im = ax.imshow(spray_probability, cmap='hot', vmin=0, vmax=1,
               extent=extent_mm, aspect='equal')
ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
ax.set_title('Spray Presence Probability Map')
ax.set_xlim(X_LIM); ax.set_ylim(Y_LIM)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
cb.set_label('Spray Presence Probability')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_spray_probability_map.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_spray_probability_map.{SAVE_FORMAT}")


# FIGURE 6: Spray boundary contours on mean image

fig, ax = plt.subplots(figsize=(5, 7))
ax.imshow(mean_image.astype(np.uint8), cmap='gray', extent=extent_mm, aspect='equal')

# Mean-image binary contour
contours_mean, _ = cv2.findContours(mean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours_mean:
    cnt_mm = contour_to_mm(cnt)
    ax.plot(cnt_mm[:, 0, 0], cnt_mm[:, 0, 1], 'c-', linewidth=1.5,
            label='Mean image (Otsu)')

# Probability contours
for prob_val, color, style, lbl in [(0.1, 'b', '--', '$P = 0.1$'),
                                     (0.5, 'lime', '-', '$P = 0.5$'),
                                     (0.9, 'r', '-', '$P = 0.9$')]:
    prob_binary = (spray_probability >= prob_val).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(prob_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        cnt_mm = contour_to_mm(cnt)
        ax.plot(cnt_mm[:, 0, 0], cnt_mm[:, 0, 1], color=color, linestyle=style,
                linewidth=1.0, alpha=0.8, label=lbl)

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc='upper right', framealpha=0.8, edgecolor='white')
ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
ax.set_title('Spray Boundary Detection')
ax.set_xlim(X_LIM); ax.set_ylim(Y_LIM)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_spray_boundary.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_spray_boundary.{SAVE_FORMAT}")


# FIGURE 7: Spray dimensions annotated

fig, ax = plt.subplots(figsize=(5.5, 7.5))
ax.imshow(mean_image.astype(np.uint8), cmap='gray', extent=extent_mm, aspect='equal')

for cnt in contours_mean:
    cnt_mm = contour_to_mm(cnt)
    ax.plot(cnt_mm[:, 0, 0], cnt_mm[:, 0, 1], 'c-', linewidth=1.5)

spray_rows = np.where(mean_binary.max(axis=1) > 0)[0]
spray_cols = np.where(mean_binary.max(axis=0) > 0)[0]

if len(spray_rows) > 0 and len(spray_cols) > 0:
    # Physical coordinates of spray extents
    y_top_phys = px_to_mm_y(spray_rows[0])     # top of spray (positive)
    y_bot_phys = px_to_mm_y(spray_rows[-1])     # bottom of spray
    x_left_phys = px_to_mm_x(spray_cols[0])
    x_right_phys = px_to_mm_x(spray_cols[-1])
    y_mid_phys = (y_top_phys + y_bot_phys) / 2
    x_mid_phys = (x_left_phys + x_right_phys) / 2

    # Width arrow at widest row
    max_w_left_phys = px_to_mm_x(left_boundary[max_w_row])
    max_w_right_phys = px_to_mm_x(right_boundary[max_w_row])
    max_w_y_phys = px_to_mm_y(max_w_row)

    ax.annotate('', xy=(max_w_right_phys, max_w_y_phys),
                xytext=(max_w_left_phys, max_w_y_phys),
                arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5))
    ax.text(x_mid_phys, max_w_y_phys + 1.0,
            f'$W_{{max}}$ = {max_width_mm:.1f} mm',
            ha='center', va='bottom', color='yellow', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Height arrow
    x_arrow_phys = x_right_phys + 2.0
    if x_arrow_phys > x_right_mm - 1:
        x_arrow_phys = x_left_phys - 2.0
    ax.annotate('', xy=(x_arrow_phys, y_top_phys),
                xytext=(x_arrow_phys, y_bot_phys),
                arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5))
    ax.text(x_arrow_phys + 0.8, y_mid_phys,
            f'$H$ = {spray_height_phys:.1f} mm',
            ha='left', va='center', color='yellow', fontsize=12, rotation=90,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Area text
    ax.text(0.02, 0.02, f'$A$ = {mean_area_mm2:.1f} mm$^2$',
            transform=ax.transAxes, ha='left', va='bottom', color='white', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
ax.set_title('Spray Dimensions (Mean Image, Otsu)')
ax.set_xlim(X_LIM); ax.set_ylim(Y_LIM)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_spray_dimensions.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_spray_dimensions.{SAVE_FORMAT}")


# FIGURE 8: Width profile

fig, ax = plt.subplots(figsize=(5, 6))
valid = width_profile_mm > 0
ax.plot(width_profile_mm[valid], y_phys[valid], 'b-', linewidth=1.0)
ax.set_xlabel('Spray Width (mm)')
ax.set_ylabel(ylabel_str)
ax.set_title('Spray Width Profile (Mean Image, Otsu)')
ax.set_ylim(Y_LIM)
# Y already increases upward naturally (positive at top)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_width_profile.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_width_profile.{SAVE_FORMAT}")


# FIGURE 9: Per-frame area time series

fig, ax = plt.subplots(figsize=(7, 3.5))
frame_nums = np.arange(1, n_frames + 1)
ax.plot(frame_nums, areas_mm2, 'b-o', markersize=3, linewidth=0.8, label='Instantaneous')
ax.axhline(np.mean(areas_mm2), color='r', linestyle='--', linewidth=1.2,
           label=f'Mean = {np.mean(areas_mm2):.1f} mm$^2$')
ax.fill_between(frame_nums,
                np.mean(areas_mm2) - np.std(areas_mm2),
                np.mean(areas_mm2) + np.std(areas_mm2),
                alpha=0.15, color='r',
                label=f'$\\pm 1\\sigma$ = {np.std(areas_mm2):.1f} mm$^2$')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Spray Area (mm$^2$)')
ax.set_title('Spray-Occupied Area per Frame')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_area_timeseries.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_area_timeseries.{SAVE_FORMAT}")


# FIGURE 10: Per-frame width & height time series

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax1.plot(frame_nums, widths_mm, 'b-o', markersize=3, linewidth=0.8)
ax1.axhline(np.mean(widths_mm), color='r', linestyle='--', linewidth=1.2,
            label=f'Mean = {np.mean(widths_mm):.1f} mm')
ax1.set_ylabel('Spray Width (mm)')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Spray Width & Height per Frame')

ax2.plot(frame_nums, heights_mm, 'g-o', markersize=3, linewidth=0.8)
ax2.axhline(np.mean(heights_mm), color='r', linestyle='--', linewidth=1.2,
            label=f'Mean = {np.mean(heights_mm):.1f} mm')
ax2.set_xlabel('Frame Number')
ax2.set_ylabel('Spray Height (mm)')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_width_height_timeseries.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_width_height_timeseries.{SAVE_FORMAT}")


# FIGURE 11: Instantaneous boundary overlay (3 sample frames)

sample_indices = [0, n_frames // 2, n_frames - 1]
fig, axes = plt.subplots(1, 3, figsize=(12, 7))
for ax, idx in zip(axes, sample_indices):
    raw = all_images[idx].astype(np.uint8)
    rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(binary_masks[idx], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, (0, 255, 255), 1)
    ax.imshow(rgb, extent=extent_mm, aspect='equal')
    ax.set_title(f'Frame #{idx+1}\n$A$ = {areas_mm2[idx]:.1f} mm$^2$')
    ax.set_xlabel(xlabel_str); ax.set_ylabel(ylabel_str)
    ax.set_xlim(X_LIM); ax.set_ylim(Y_LIM)
plt.suptitle('Instantaneous Spray Boundary Detection', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"fig_instantaneous_boundaries.{SAVE_FORMAT}"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: fig_instantaneous_boundaries.{SAVE_FORMAT}")


# Save CSV

csv_path = OUTPUT_DIR / "spray_results.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(["COORDINATE SYSTEM"])
    writer.writerow(["Origin", "Nozzle exit (fixed)"])
    writer.writerow(["X_origin_mm_from_left", X_ORIGIN_FROM_LEFT_MM])
    writer.writerow(["Y_origin_mm_from_top", Y_ORIGIN_FROM_TOP_MM])
    writer.writerow(["X_origin_px", f"{x_origin_px:.1f}"])
    writer.writerow(["Nozzle_row_px", f"{nozzle_row_px:.1f}"])
    writer.writerow(["Y_direction", "Positive upward"])
    writer.writerow([])

    writer.writerow(["TIME-AVERAGED RESULTS (from mean image)"])
    writer.writerow(["Metric", "Value_px", "Value_mm", "Units"])
    writer.writerow(["Spray_area", mean_area_px, f"{mean_area_mm2:.2f}", "mm2"])
    writer.writerow(["Max_width", f"{width_profile_px[max_w_row]:.0f}", f"{max_width_mm:.2f}", "mm"])
    writer.writerow(["Height", mean_height_px, f"{spray_height_phys:.2f}", "mm"])
    writer.writerow(["Spray_top_y", mean_top_row, f"{spray_y_top_mm:.2f}", "mm"])
    writer.writerow(["Spray_bottom_y", mean_bot_row, f"{spray_y_bot_mm:.2f}", "mm"])
    writer.writerow([])

    writer.writerow(["PER-FRAME RESULTS"])
    writer.writerow(["Frame", "Filename", "Area_px", "Area_mm2",
                     "Width_px", "Width_mm", "Height_px", "Height_mm"])
    for i in range(n_frames):
        writer.writerow([i+1, image_files[i].name,
                         areas_px[i], f"{areas_mm2[i]:.4f}",
                         widths_px[i], f"{widths_mm[i]:.4f}",
                         heights_px[i], f"{heights_mm[i]:.4f}"])
    writer.writerow([])
    writer.writerow(["PER-FRAME STATISTICS"])
    writer.writerow(["Metric", "Mean", "Std", "Units"])
    writer.writerow(["Area", f"{np.mean(areas_mm2):.2f}", f"{np.std(areas_mm2):.2f}", "mm2"])
    writer.writerow(["Width", f"{np.mean(widths_mm):.2f}", f"{np.std(widths_mm):.2f}", "mm"])
    writer.writerow(["Height", f"{np.mean(heights_mm):.2f}", f"{np.std(heights_mm):.2f}", "mm"])
    writer.writerow([])
    writer.writerow(["PARAMETERS"])
    writer.writerow(["Pixel_to_mm", PIXEL_TO_MM])
    writer.writerow(["Threshold_method", "Otsu (on mean background-subtracted image)"])
    writer.writerow(["Threshold_value", f"{otsu_threshold:.0f}"])
    writer.writerow(["Background_method", "Edge sampling + linear interpolation from mean image"])
    writer.writerow(["Edge_margin_px", edge_margin])
    writer.writerow(["Bar_interp_method", "Vertical linear interpolation across bar band"])
    writer.writerow(["Morph_close", "7x7 ellipse, 2 iterations"])
    writer.writerow(["Morph_open", "7x7 ellipse, 2 iterations"])
    writer.writerow(["Min_blob_area", f"{min_blob_area} pixels"])
    writer.writerow(["Num_frames", n_frames])

print(f"Saved: {csv_path}")


# Save .npz data for 3D reconstruction

# Coordinate grids (physical mm, nozzle = origin)
x_grid_mm = np.array([px_to_mm_x(c) for c in range(w)])
y_grid_mm = np.array([px_to_mm_y(r) for r in range(h)])

# Extract Otsu contour coordinates in mm
contours_for_save, _ = cv2.findContours(mean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if contours_for_save:
    # Use the largest contour (main spray plume)
    largest_cnt = max(contours_for_save, key=cv2.contourArea)
    cnt_mm_save = contour_to_mm(largest_cnt)
    otsu_contour_x = cnt_mm_save[:, 0, 0]
    otsu_contour_y = cnt_mm_save[:, 0, 1]
else:
    otsu_contour_x = np.array([])
    otsu_contour_y = np.array([])

npz_path = OUTPUT_DIR / "section_data_otsu.npz"
np.savez(npz_path,
    # Contour boundary (Otsu on mean image)
    contour_x_mm=otsu_contour_x,
    contour_y_mm=otsu_contour_y,
    # 2D fields
    spray_probability=spray_probability,
    binary_mask=mean_binary,
    mean_subtracted=mean_subtracted.astype(np.float32),
    mean_image=mean_image.astype(np.float32),
    background=background.astype(np.float32),
    # Coordinate grids
    x_grid_mm=x_grid_mm.astype(np.float32),
    y_grid_mm=y_grid_mm.astype(np.float32),
    # Spray metrics
    area_mm2=np.float64(mean_area_mm2),
    max_width_mm=np.float64(max_width_mm),
    height_mm=np.float64(spray_height_phys),
    spray_top_y_mm=np.float64(spray_y_top_mm),
    spray_bot_y_mm=np.float64(spray_y_bot_mm),
    # Width profile
    width_profile_mm=width_profile_mm.astype(np.float32),
    y_profile_mm=y_phys.astype(np.float32),
    # Per-frame statistics
    frame_areas_mm2=areas_mm2,
    frame_widths_mm=widths_mm,
    frame_heights_mm=heights_mm,
    # Parameters
    otsu_threshold=np.float64(otsu_threshold),
    pixel_to_mm=np.float64(PIXEL_TO_MM),
    x_origin_from_left_mm=np.float64(X_ORIGIN_FROM_LEFT_MM),
    y_origin_from_top_mm=np.float64(Y_ORIGIN_FROM_TOP_MM),
    n_frames=np.int32(n_frames),
)
print(f"Saved: {npz_path}")
print("  -> Contains: contour, probability map, binary mask, width profile,")
print("     per-frame stats, coordinate grids (for 3D reconstruction)")

print("\nAll done!")

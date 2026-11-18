"""Picks colors for pie chart slices using the Colorgorical method (Gramazio et al. 2017).

Main function (this is what run.py should call):
    choose_colors(n_colors, ...)  ->  ["#1f77b4", "#cbe013", ...]

How it works:
    1. Make a grid of candidate colors.
    2. Start the palette with the seed color(s), or a random candidate.
    3. Score every candidate and add the best one. Repeat until we have n_colors.
       score = distance_weight   * how different it looks from the closest palette color
             + preference_weight * how nice it looks next to the palette colors

Test by itself:
    python color_choice.py 6 --random-seed 1
"""

import argparse
from itertools import combinations

import numpy as np

# Default settings (all can be overridden when calling choose_colors)
DISTANCE_WEIGHT = 1.0       # how much we care that colors are easy to tell apart
PREFERENCE_WEIGHT = 0.5     # how much we care that colors look nice together
LIGHTNESS_RANGE = (25, 85)  # 0 = black, 100 = white; skip near-black and near-white
GRID_STEP = 5               # spacing between candidate colors (smaller = more candidates, slower)

# Standard sRGB constants used for the color conversions
WHITE_POINT = np.array([0.95047, 1.00000, 1.08883])
RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
XYZ_TO_RGB = np.linalg.inv(RGB_TO_XYZ)


# ---------------------------------------------------------------------------
# Color conversions
# Lab is a color space where distances match how different colors look to people:
#   L = lightness (0-100), a = green<->red, b = blue<->yellow
# These functions work on one color [L, a, b] or an array of many colors.
# ---------------------------------------------------------------------------
def split_lab(lab):
    """Splits Lab color(s) into separate L, a and b values."""
    lab = np.asarray(lab, dtype=float)
    return lab[..., 0], lab[..., 1], lab[..., 2]


def hex_to_lab(hex_color):
    """Converts a hex string like "#1f77b4" into a Lab color [L, a, b]."""
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:  # short form, e.g. "f80" -> "ff8800"
        hex_color = "".join(char * 2 for char in hex_color)
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")

    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]) / 255
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (RGB_TO_XYZ @ linear_rgb) / WHITE_POINT

    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    fx, fy, fz = f
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def lab_to_linear_rgb(lab):
    """Converts Lab color(s) to linear RGB; values outside 0-1 mean the color can't be shown on screen."""
    L, a, b = split_lab(lab)
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], axis=-1)

    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * WHITE_POINT
    return xyz @ XYZ_TO_RGB.T


def lab_to_hex(lab):
    """Converts a Lab color [L, a, b] into a hex string like "#1f77b4"."""
    linear_rgb = np.clip(lab_to_linear_rgb(lab), 0, 1)
    rgb = np.where(linear_rgb <= 0.0031308, 12.92 * linear_rgb, 1.055 * linear_rgb ** (1 / 2.4) - 0.055)
    red, green, blue = [int(round(value * 255)) for value in rgb]
    return f"#{red:02x}{green:02x}{blue:02x}"


def lab_to_lch(lab):
    """Converts Lab color(s) to [lightness, chroma (how colorful), hue angle in degrees]."""
    L, a, b = split_lab(lab)
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    return np.stack([L, chroma, hue], axis=-1)


# ---------------------------------------------------------------------------
# The two scores
# ---------------------------------------------------------------------------
def ciede2000(lab1, lab2):
    """Returns how different two colors look (about 2 = barely noticeable, 10+ = clearly different)."""
    # Standard CIEDE2000 formula (Sharma et al. 2005). It's long, but you shouldn't need
    # to change it -- test_color_choice.py checks it against published values.
    L1, a1, b1 = split_lab(lab1)
    L2, a2, b2 = split_lab(lab2)

    # adjust a* so grayish colors are handled better
    avg_chroma = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(avg_chroma ** 7 / (avg_chroma ** 7 + 25 ** 7)))
    a1, a2 = (1 + G) * a1, (1 + G) * a2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    h1 = np.degrees(np.arctan2(b1, a1)) % 360
    h2 = np.degrees(np.arctan2(b2, a2)) % 360
    has_gray = (C1 * C2) == 0

    # differences in lightness, chroma and hue
    dL = L2 - L1
    dC = C2 - C1
    dh = (h2 - h1 + 180) % 360 - 180  # shortest way around the color wheel
    dh = np.where(has_gray, 0, dh)
    dH = 2 * np.sqrt(C1 * C2) * np.sin(np.radians(dh / 2))

    # averages
    avg_L = (L1 + L2) / 2
    avg_C = (C1 + C2) / 2
    avg_h = np.where(np.abs(h1 - h2) <= 180, (h1 + h2) / 2, (h1 + h2 + 360) / 2) % 360
    avg_h = np.where(has_gray, h1 + h2, avg_h)

    # weighting terms
    T = (1 - 0.17 * np.cos(np.radians(avg_h - 30)) + 0.24 * np.cos(np.radians(2 * avg_h))
         + 0.32 * np.cos(np.radians(3 * avg_h + 6)) - 0.20 * np.cos(np.radians(4 * avg_h - 63)))
    S_L = 1 + 0.015 * (avg_L - 50) ** 2 / np.sqrt(20 + (avg_L - 50) ** 2)
    S_C = 1 + 0.045 * avg_C
    S_H = 1 + 0.015 * avg_C * T
    rotation = 30 * np.exp(-((avg_h - 275) / 25) ** 2)
    R_T = -np.sin(np.radians(2 * rotation)) * 2 * np.sqrt(avg_C ** 7 / (avg_C ** 7 + 25 ** 7))

    return np.sqrt((dL / S_L) ** 2 + (dC / S_C) ** 2 + (dH / S_H) ** 2
                   + R_T * (dC / S_C) * (dH / S_H))


def fade_to_neutral(score, strength):
    """Pulls a 0-1 score toward 0.5 when strength is low (used for grays, which have no real hue)."""
    return 0.5 + (score - 0.5) * strength


def pair_preference(lab1, lab2):
    """Returns how nice two colors look together, from 0 to 1 (approximation of Schloss & Palmer 2011)."""
    # People tend to like pairs with similar hues, cool colors, and strong lightness contrast.
    L1, chroma1, hue1 = np.moveaxis(lab_to_lch(lab1), -1, 0)
    L2, chroma2, hue2 = np.moveaxis(lab_to_lch(lab2), -1, 0)

    # how much each color's hue "counts": 0 for gray, 1 for colorful (chroma 30+)
    hue_strength1 = np.minimum(chroma1 / 30, 1)
    hue_strength2 = np.minimum(chroma2 / 30, 1)

    # 1 = same hue, 0 = opposite sides of the color wheel
    hue_similarity = (1 + np.cos(np.radians(hue1 - hue2))) / 2
    hue_similarity = fade_to_neutral(hue_similarity, np.minimum(hue_strength1, hue_strength2))

    # 0 = warmest (red-orange, ~50 degrees), 1 = coolest (blue-cyan, ~230 degrees)
    coolness1 = fade_to_neutral((1 - np.cos(np.radians(hue1 - 50))) / 2, hue_strength1)
    coolness2 = fade_to_neutral((1 - np.cos(np.radians(hue2 - 50))) / 2, hue_strength2)
    coolness = (coolness1 + coolness2) / 2

    # 0 = same lightness, 1 = black vs white
    lightness_contrast = np.abs(L1 - L2) / 100

    return (hue_similarity + coolness + lightness_contrast) / 3


# ---------------------------------------------------------------------------
# Picking the palette
# ---------------------------------------------------------------------------
def build_candidate_space(lightness_range=LIGHTNESS_RANGE, hue_range=None):
    """Returns every color on a Lab grid that can be shown on screen and fits the lightness/hue limits."""
    L_values = np.arange(lightness_range[0], lightness_range[1] + 0.001, GRID_STEP)
    ab_values = np.arange(-128, 128.001, GRID_STEP)
    candidates = np.array([[L, a, b] for L in L_values for a in ab_values for b in ab_values])

    # keep only colors a normal monitor can display
    rgb = lab_to_linear_rgb(candidates)
    on_screen = np.all((rgb >= -1e-6) & (rgb <= 1 + 1e-6), axis=1)
    candidates = candidates[on_screen]

    # optional hue limit, e.g. (180, 300) for blues; (330, 30) wraps around through red
    if hue_range is not None:
        _, chroma, hue = lab_to_lch(candidates).T
        start, end = hue_range[0] % 360, hue_range[1] % 360
        if start <= end:
            in_range = (hue >= start) & (hue <= end)
        else:
            in_range = (hue >= start) | (hue <= end)
        candidates = candidates[in_range & (chroma >= 5)]  # grays have no hue, so drop them

    if len(candidates) == 0:
        raise ValueError("No candidate colors left; loosen the lightness/hue limits.")
    return candidates


def scale_0_to_1(values):
    """Rescales values so the smallest becomes 0 and the largest becomes 1."""
    spread = values.max() - values.min()
    if spread == 0:
        return np.zeros_like(values)
    return (values - values.min()) / spread


def pick_next_color(palette, candidates, distance_weight, preference_weight):
    """Scores every candidate against the current palette and returns the best one."""
    # one row per palette color, one column per candidate
    distances = np.array([ciede2000(color, candidates) for color in palette])
    preferences = np.array([pair_preference(color, candidates) for color in palette])

    closest_distance = distances.min(axis=0)       # how far it is from the most similar palette color
    average_preference = preferences.mean(axis=0)  # how nice it looks with the whole palette

    # skip colors already in the palette, then score the rest
    available = closest_distance > 1e-6
    score = (distance_weight * scale_0_to_1(closest_distance[available])
             + preference_weight * scale_0_to_1(average_preference[available]))
    return candidates[available][np.argmax(score)]


def choose_colors(n_colors, seed_colors=None, distance_weight=DISTANCE_WEIGHT,
                  preference_weight=PREFERENCE_WEIGHT, lightness_range=LIGHTNESS_RANGE,
                  hue_range=None, random_seed=None):
    """Picks n_colors hex colors (one per pie slice), keeping any seed_colors at the front."""
    seed_colors = seed_colors or []
    if n_colors < 1:
        raise ValueError("n_colors must be at least 1")
    if len(seed_colors) > n_colors:
        raise ValueError(f"Got {len(seed_colors)} seed colors but only asked for {n_colors} colors")

    candidates = build_candidate_space(lightness_range, hue_range)
    if n_colors - len(seed_colors) > len(candidates):
        raise ValueError("Not enough candidate colors; loosen the lightness/hue limits.")

    # start with the seed colors, or one random candidate if there are none
    palette = [hex_to_lab(color) for color in seed_colors]
    if not palette:
        rng = np.random.default_rng(random_seed)
        palette.append(candidates[rng.integers(len(candidates))])

    while len(palette) < n_colors:
        palette.append(pick_next_color(palette, candidates, distance_weight, preference_weight))

    return [lab_to_hex(color) for color in palette]


# ---------------------------------------------------------------------------
# Describing a palette (for output and comparing palettes -- doesn't affect picking)
# ---------------------------------------------------------------------------
# Everyday color names, with hex values based on the XKCD color survey (public domain).
# Add or remove names here to change what gets printed.
COLOR_NAMES = {
    "black": "#000000", "dark gray": "#363737", "gray": "#929591", "light gray": "#d8dcd6",
    "white": "#ffffff",
    "dark red": "#840000", "red": "#e50000", "maroon": "#650021", "salmon": "#ff796c",
    "orange": "#f97306", "light orange": "#fdaa48", "dark brown": "#653700", "brown": "#875f42",
    "light brown": "#ad8150", "tan": "#d1b26f", "beige": "#e6daa6", "gold": "#dbb40c",
    "yellow": "#ffff14", "olive": "#6e750e", "dark olive": "#373e02", "lime": "#aaff32",
    "light green": "#96f97b", "green": "#15b01a", "bright green": "#01ff07", "dark green": "#033500",
    "turquoise": "#06c2ac", "teal": "#029386", "dark teal": "#014d4e", "cyan": "#00ffff",
    "light blue": "#95d0fc", "sky blue": "#75bbfd", "blue": "#0343df", "dark blue": "#00035b",
    "indigo": "#380282", "violet": "#9a0eea", "purple": "#7e1e9c", "dark purple": "#35063e",
    "lavender": "#c79fef", "magenta": "#c20078", "hot pink": "#ff028d", "pink": "#ff81c0",
    "light pink": "#ffd1df",
}


def color_name(hex_color):
    """Returns the closest everyday name for a hex color, e.g. "#1f77b4" -> "blue"."""
    names = list(COLOR_NAMES)
    name_labs = np.array([hex_to_lab(COLOR_NAMES[name]) for name in names])
    distances = ciede2000(hex_to_lab(hex_color), name_labs)
    return names[int(np.argmin(distances))]


def palette_stats(hex_colors):
    """Returns summary numbers for a palette so different palettes can be compared."""
    # min_delta_e:          difference between the two MOST similar colors (higher = easier to tell apart)
    # mean_delta_e:         average difference over every pair of colors
    # mean_pair_preference: average "looks nice together" score, 0-1
    if len(hex_colors) < 2:
        return {"min_delta_e": None, "mean_delta_e": None, "mean_pair_preference": None}

    labs = [hex_to_lab(color) for color in hex_colors]
    pairs = list(combinations(labs, 2))  # every pair of colors, each counted once
    distances = [float(ciede2000(color1, color2)) for color1, color2 in pairs]
    preferences = [float(pair_preference(color1, color2)) for color1, color2 in pairs]

    return {
        "min_delta_e": min(distances),
        "mean_delta_e": sum(distances) / len(distances),
        "mean_pair_preference": sum(preferences) / len(preferences),
    }


def main():
    """Command line entry point: picks colors and prints them with names and stats (no chart)."""
    parser = argparse.ArgumentParser(description="Pick Colorgorical-style colors for a pie chart.")
    parser.add_argument("n_colors", type=int, help="number of colors (pie slices)")
    parser.add_argument("--seed-colors", nargs="*", help='starting hex colors, e.g. "#1f77b4"')
    parser.add_argument("--distance-weight", type=float, default=DISTANCE_WEIGHT)
    parser.add_argument("--preference-weight", type=float, default=PREFERENCE_WEIGHT)
    parser.add_argument("--lightness", type=float, nargs=2, metavar=("MIN", "MAX"), default=LIGHTNESS_RANGE)
    parser.add_argument("--hue-range", type=float, nargs=2, metavar=("START", "END"))
    parser.add_argument("--random-seed", type=int)
    args = parser.parse_args()

    colors = choose_colors(
        args.n_colors,
        seed_colors=args.seed_colors,
        distance_weight=args.distance_weight,
        preference_weight=args.preference_weight,
        lightness_range=args.lightness,
        hue_range=args.hue_range,
        random_seed=args.random_seed,
    )

    print("Colors:")
    for color in colors:
        print(f"  {color}  {color_name(color)}")

    print("Stats:")
    for stat, value in palette_stats(colors).items():
        print(f"  {stat}: {value:.2f}" if value is not None else f"  {stat}: n/a")


if __name__ == "__main__":
    main()

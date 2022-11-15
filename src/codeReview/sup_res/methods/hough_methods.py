import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line

from src.codeReview.utils import get_bestfit


# Utility functions
def make_image(Idxs, h, min_h, max_h, hough_scale, len_h):
    # (the below are) minimal angles to find all points
    # np.arctan(2/len_h), np.arctan(2/int((hist.Close.max() - m + 1) * (1/hough_scale)))
    max_size = int(np.ceil(2 / np.tan(np.pi / (360 * 5))))  # ~1146
    m, tested_angles = min_h, np.linspace(-np.pi / 2, np.pi / 2,
                                            360 * 5)  # degree of precision from 90 to 270 degrees with 360*5 increments
    height = int((max_h - m + 0.01) * (1 / hough_scale))
    mx = min(max_size, height)
    scl = (1 / hough_scale) * mx / height
    image = np.zeros((mx, len_h))  # in rows, columns or y, x image format
    for x in Idxs:
        image[int((h[x] - m) * scl), x] = 255

    return image, tested_angles, scl, m

def find_line_pts(Idxs, x0, y0, x1, y1, h, fltpct):
    s = (y0 - y1) / (x0 - x1)
    i, dnm = y0 - s * x0, np.sqrt(1 + s * s)
    dist = [(np.abs(i + s * x - h[x]) / dnm, x) for x in Idxs]
    dist.sort()  # (key=lambda val: val[0])
    pts, res = [], None
    for x in range(len(dist)):
        pts.append((dist[x][1], h[dist[x][1]]))
        if len(pts) < 3: continue
        r = get_bestfit(pts)
        if r[3] > fltpct:
            pts = pts[:-1]
            break
        res = r
    pts = [x for x, _ in pts]
    pts.sort()

    return pts, res

def hough_points(pts, width, height, thetas):
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    # Vote in the hough accumulator
    for i in range(len(pts)):
        x, y = pts[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_len
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


# --- Actual Methods

def houghpt(Idxs, h, fltpct, min_h, max_h, hough_scale, hough_prob_iter, len_h):
        max_size = int(np.ceil(2 / np.tan(np.pi / (360 * 5))))  # ~1146
        m, tested_angles = min_h, np.linspace(-np.pi / 2, np.pi / 2,
                                              360 * 5)  # degree of precision from 90 to 270 degrees with 360*5 increments
        height = int((max_h - m + 1) * (1 / hough_scale))
        mx = min(max_size, height)
        scl = (1 / hough_scale) * mx / height
        acc, theta, d = hough_points([(x, int((h[x] - m) * scl)) for x in Idxs], mx, len_h,
                                     np.linspace(-np.pi / 2, np.pi / 2, 360 * 5))
        origin, lines = np.array((0, len_h)), []
        for x, y in np.argwhere(acc >= 3):
            dist, angle = d[x], theta[y]
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            y0, y1 = y0 / scl + m, y1 / scl + m
            pts, res = find_line_pts(Idxs, 0, y0, len_h, y1, h, fltpct)
            if len(pts) >= 3: lines.append((pts, res))

        return lines


def hough(Idxs, h, fltpct, min_h, max_h, hough_scale, hough_prob_iter, len_h):
    image, tested_angles, scl, m = make_image(Idxs, h, min_h, max_h, hough_scale, len_h)
    hl, theta, d = hough_line(image, theta=tested_angles)
    origin, lines = np.array((0, image.shape[1])), []
    for pts, angle, dist in zip(*hough_line_peaks(hl, theta, d, threshold=2)):  # > threshold
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        y0, y1 = y0 / scl + m, y1 / scl + m
        pts, res = find_line_pts(Idxs, 0, y0, image.shape[1], y1, h, fltpct)
        if len(pts) >= 3: lines.append((pts, res))

    return lines


def prob_hough(Idxs, h, fltpct, min_h, max_h, hough_scale, hough_prob_iter, len_h):
    image, tested_angles, scl, m = make_image(Idxs, h, min_h, max_h, hough_scale, len_h)
    lines = []
    for x in range(hough_prob_iter):
        lines.extend(probabilistic_hough_line(
            image, threshold=2, theta=tested_angles, line_length=0,
            line_gap=int(np.ceil(np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1])))))
        )
    l = []
    for (x0, y0), (x1, y1) in lines:
        if x0 == x1: continue
        if x1 < x0: (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
        y0, y1 = y0 / scl + m, y1 / scl + m
        pts, res = find_line_pts(Idxs, x0, y0, x1, y1, h, fltpct)
        if len(pts) >= 3:
            l.append((pts, res))

    return l
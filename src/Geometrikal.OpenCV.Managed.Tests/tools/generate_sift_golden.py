#!/usr/bin/env python3
"""Golden-data generator for the managed SIFT port.

Runs cv2 5.0.0 ``SIFT_create`` (``detect``, ``compute`` and ``detectAndCompute``) over a
seeded battery of synthetic textured images and writes ``TestData/sift_golden.json``.
``Features/SiftGoldenTests.cs`` rebuilds each image with the same integer formula, runs
the port and compares keypoints and descriptors.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>``. numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set). The port reproduces
   the 128-bit baseline dispatch (4 float lanes, no FMA).
 * IPP and OpenCL are disabled in the script. One thread is used; the keypoint order
   does not depend on the thread count because removeDuplicatedSorted sorts the list.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_sift_golden.py

Descriptors of cases with more than 1500 keypoints are stored as SHA-256 (``descriptorsSha``).

Image formula (``make_plane``), all integer arithmetic on uint32 / int64:
 * hash(x, y, s) = h ^= h >> 13; h *= 0x5BD1E995; h ^= h >> 15, starting from
   h = (x * 73856093) ^ (y * 19349663) ^ (s * 83492791), modulo 2^32.
 * bilinear(cell, s): corner values hash(gx, gy, s) & 255 on a grid of ``cell`` pixels,
   interpolated with integer weights and a right shift by log2(cell^2).
 * value = (2 * bilinear(8, seed) + bilinear(32, seed + 1)) // 3
           + (hash(x, y, seed + 2) & 15) - 8, then 6 rectangles, clipped to [0, 255].
 * Rectangle k: r = hash(k, 0, seed + 3); x0 = r % w, y0 = (r >> 8) % h,
   rw = 4 + (r >> 16) % 24, rh = 4 + (r >> 24) % 24, value = hash(k, 1, seed + 3) & 255.
 * BGR images use seeds seed, seed + 10, seed + 20 for the three channels.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_sift_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

M32 = 0xFFFFFFFF


def hash32(x, y, s):
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    m = np.uint64(M32)
    h = ((x * np.uint64(73856093)) & m) ^ ((y * np.uint64(19349663)) & m) ^ np.uint64((s * 83492791) & M32)
    h = h ^ (h >> np.uint64(13))
    h = (h * np.uint64(0x5BD1E995)) & m
    h = h ^ (h >> np.uint64(15))
    return h.astype(np.int64)


def bilinear(w, h, cell, s, shift):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64)
    gx, gy = xs // cell, ys // cell
    fx, fy = xs % cell, ys % cell
    a = hash32(gx, gy, s) & 255
    b = hash32(gx + 1, gy, s) & 255
    c = hash32(gx, gy + 1, s) & 255
    d = hash32(gx + 1, gy + 1, s) & 255
    v = a * (cell - fx) * (cell - fy) + b * fx * (cell - fy) + c * (cell - fx) * fy + d * fx * fy
    return v >> shift


def make_plane(w, h, seed):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64)
    v = (2 * bilinear(w, h, 8, seed, 6) + bilinear(w, h, 32, seed + 1, 10)) // 3
    v = v + (hash32(xs, ys, seed + 2) & 15) - 8
    for k in range(6):
        r = int(hash32(k, 0, seed + 3))
        x0, y0 = r % w, (r >> 8) % h
        rw, rh = 4 + (r >> 16) % 24, 4 + (r >> 24) % 24
        val = int(hash32(k, 1, seed + 3)) & 255
        v[y0:y0 + rh, x0:x0 + rw] = val
    return np.clip(v, 0, 255).astype(np.uint8)


def make_image(w, h, cn, seed):
    if cn == 1:
        return make_plane(w, h, seed)
    return np.ascontiguousarray(np.dstack([make_plane(w, h, seed + 10 * c) for c in range(cn)]))


def make_mask(w, h):
    m = np.zeros((h, w), np.uint8)
    m[: (3 * h) // 4, w // 4:] = 255
    return m


def kp_list(kps):
    return [[float(np.float32(k.pt[0])), float(np.float32(k.pt[1])), float(np.float32(k.size)),
             float(np.float32(k.angle)), float(np.float32(k.response)), int(k.octave), int(k.class_id)]
            for k in kps]


def desc_b64(d):
    if d is None:
        return ""
    return base64.b64encode(np.ascontiguousarray(d).tobytes()).decode("ascii")


CASES = []


def add(name, w, h, cn=1, seed=1, mask=False, mode="detectAndCompute", nfeatures=0, layers=3,
        contrast=0.04, edge=10.0, sigma=1.6, dtype=cv2.CV_32F, precise=True):
    img = make_image(w, h, cn, seed)
    m = make_mask(w, h) if mask else None
    sift = cv2.SIFT_create(nfeatures, layers, contrast, edge, sigma, dtype, precise)
    case = {
        "name": name, "width": w, "height": h, "channels": cn, "seed": seed, "mask": mask,
        "mode": mode, "nfeatures": nfeatures, "nOctaveLayers": layers, "contrastThreshold": contrast,
        "edgeThreshold": edge, "sigma": sigma, "descriptorType": "u8" if dtype == cv2.CV_8U else "f32",
        "preciseUpscale": precise, "imageSha": hashlib.sha256(img.tobytes()).hexdigest(),
    }
    if mode == "detect":
        kps = sift.detect(img, m)
        desc = None
    elif mode in ("compute", "computeOctave0"):
        # Keypoints from a detector with other parameters, then compute with this one.
        src = cv2.SIFT_create(0, 3, 0.03, 10.0, 1.6).detect(img, None)
        if mode == "computeOctave0":
            # Without octave -1 keypoints, compute builds no upscaled level (firstOctave = 0).
            src = [k for k in src if (k.octave & 255) != 255]
        case["inputKeypoints"] = kp_list(src)
        kps, desc = sift.compute(img, src)
    else:
        kps, desc = sift.detectAndCompute(img, m)
    case["keypoints"] = kp_list(kps)
    if desc is not None and len(kps) > 1500:
        case["descriptors"] = ""
        case["descriptorsSha"] = hashlib.sha256(np.ascontiguousarray(desc).tobytes()).hexdigest()
    else:
        case["descriptors"] = desc_b64(desc)
    CASES.append(case)
    print(f"{name}: {len(kps)} keypoints")


add("gray_64", 64, 64, seed=1)
add("gray_96x80", 96, 80, seed=2)
add("gray_128", 128, 128, seed=3)
add("gray_200x150", 200, 150, seed=4)
add("gray_256", 256, 256, seed=5)
add("gray_512x384", 512, 384, seed=6)
add("bgr_128", 128, 128, cn=3, seed=7)
add("mask_256", 256, 256, seed=8, mask=True)
add("nfeatures50_256", 256, 256, seed=9, nfeatures=50)
add("nfeatures200_200x150", 200, 150, seed=10, nfeatures=200)
add("u8desc_256", 256, 256, seed=11, dtype=cv2.CV_8U)
add("noprecise_128", 128, 128, seed=12, precise=False)
add("params_160", 160, 160, seed=13, layers=4, contrast=0.03, edge=8.0, sigma=1.4)
add("detect_128", 128, 128, seed=14, mode="detect")
add("compute_128", 128, 128, seed=15, mode="compute")
add("compute_octave0_160", 160, 160, seed=16, mode="computeOctave0")
add("bgra_100x90", 100, 90, cn=4, seed=17)
add("tiny_24x20", 24, 20, seed=18)
add("layers5_180x140", 180, 140, seed=19, layers=5)
add("layers2_sigma2_150", 150, 150, seed=20, layers=2, sigma=2.0, contrast=0.02)
add("odd_257x131", 257, 131, seed=21)
add("gray_1024x768", 1024, 768, seed=22)

# Math kernels used by SIFT: hal::exp32f (cv2.exp), hal::fastAtan32f (cv2.phase, degrees),
# hal::magnitude32f (cv2.magnitude). Inputs are integer-valued gradients and SIFT-like weights.
rng = np.random.default_rng(2024)
mx = rng.integers(-255, 256, 3001).astype(np.float32)
my = rng.integers(-255, 256, 3001).astype(np.float32)
mx[:9] = [0, 0, 1, -1, 0, 5, -5, 5, -5]
my[:9] = [0, 1, 0, 0, -1, 5, 5, -5, -5]
mw = (-(rng.random(3001) * 12)).astype(np.float32)
MATH = {
    "x": base64.b64encode(mx.tobytes()).decode("ascii"),
    "y": base64.b64encode(my.tobytes()).decode("ascii"),
    "w": base64.b64encode(mw.tobytes()).decode("ascii"),
    "exp": base64.b64encode(cv2.exp(mw.reshape(1, -1)).ravel().tobytes()).decode("ascii"),
    "phaseDeg": base64.b64encode(cv2.phase(mx.reshape(1, -1), my.reshape(1, -1), angleInDegrees=True).ravel().tobytes()).decode("ascii"),
    "magnitude": base64.b64encode(cv2.magnitude(mx.reshape(1, -1), my.reshape(1, -1)).ravel().tobytes()).decode("ascii"),
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "sift_golden.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"opencv": cv2.__version__, "math": MATH, "cases": CASES}, f, separators=(",", ":"))
print(f"wrote {out}")

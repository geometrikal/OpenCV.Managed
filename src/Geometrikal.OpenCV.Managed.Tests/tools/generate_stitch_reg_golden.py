#!/usr/bin/env python3
"""Golden-data generator for the managed stitching registration port.

Runs the cv2 5.0.0 stitching ``detail`` matchers, motion estimators, bundle
adjusters, wave correction and graph helpers over synthetic tile sets and
writes ``TestData/stitch_reg_golden.json``.
``Stitching/RegistrationGoldenTests.cs`` replays each case.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0 built with the internal LAPACK (``Lapack: YES (Built-In
   libclapack)``). Put its directory on ``PYTHONPATH`` or set ``CV2_500_DIR``.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set).
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    CV2_500_DIR=<cv2 5.0.0 dir> python tools/generate_stitch_reg_golden.py

Battery notes:
 * Each tile stores its ORB keypoint positions and descriptors. The managed
   test builds ``ImageFeatures`` from them, so the comparison isolates the
   matchers and estimators from feature detection.
 * OpenCV ``CpuMatcher`` uses FLANN (LSH for binary descriptors). The port
   uses an exhaustive Hamming search. Each matcher case stores a Python
   reference of ``matchers.cpp`` with ``cv2.BFMatcher`` (``reference``) and the
   cv2 FLANN result (``cv2``); ``flannAgrees`` is true when both are equal.
 * The estimator, bundle adjuster and wave correction cases use the cv2
   matcher output as input. The bundle adjusters use the estimator output with
   R converted to float32 (as ``Stitcher::estimateCameraParams``); the wave
   correction cases use the bundle adjuster output.
 * Every case runs twice and the outputs must be equal, except the cv2 FLANN
   matcher: LSH uses random hash tables, so two runs can differ
   (``flannRepeatable``). The stored ``cv2`` output is the run that the
   estimator cases use.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import math
import os
import sys

if os.environ.get("CV2_500_DIR"):
    sys.path.insert(0, os.environ["CV2_500_DIR"])
os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_stitch_reg_golden.py requires cv2 5.0.0 (found {cv2.__version__})")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")
if "Built-In libclapack" not in cv2.getBuildInformation():
    sys.exit("cv2 must be built with the internal LAPACK (Lapack: YES (Built-In libclapack))")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

# Adjudicated differences (case name -> reason). Empty when the port matches every case.
SUSPECT_CASES = {}

F32 = np.float32


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


# ---------------------------------------------------------------- scene

def make_scene(seed=1, w=1600, h=1200):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    for _ in range(400):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(4, 40))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, int(rng.integers(0, 256)), -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(rng.integers(4, 40))), int(rng.integers(0, 256)), -1)
    return img


def affine_tiles(img, count=5, tw=400, th=320, step=230):
    rng = np.random.default_rng(7)
    tiles = []
    for i in range(count):
        a = math.radians(rng.uniform(-3, 3))
        s = rng.uniform(0.97, 1.03)
        ox, oy = 150 + i * step, 300 + rng.uniform(-20, 20)
        # Maps tile pixels to scene pixels; the tile is the inverse warp.
        m = np.array([[s * math.cos(a), -s * math.sin(a), ox], [s * math.sin(a), s * math.cos(a), oy]])
        tiles.append(cv2.warpAffine(img, m, (tw, th), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    return tiles


def homography_tiles(img, count=4, tw=420, th=340, step=240):
    rng = np.random.default_rng(11)
    tiles = []
    for i in range(count):
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
        ox, oy = 200 + i * step, 400
        dst = src + np.array([ox, oy], np.float32) + rng.uniform(-25, 25, (4, 2)).astype(np.float32)
        h = cv2.getPerspectiveTransform(src, dst)
        tiles.append(cv2.warpPerspective(img, h, (tw, th), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    return tiles


def rotation_tiles(img, angles, axis, f=700.0, tw=480, th=360):
    h, w = img.shape
    k_scene = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    k_tile = np.array([[f, 0, tw / 2], [0, f, th / 2], [0, 0, 1]])
    tiles = []
    for a in angles:
        c, s = math.cos(a), math.sin(a)
        if axis == "yaw":
            r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        # Maps tile pixels to scene pixels.
        hmat = k_scene @ r @ np.linalg.inv(k_tile)
        tiles.append(cv2.warpPerspective(img, hmat, (tw, th), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    return tiles


def features_of(tiles, nfeatures):
    orb = cv2.ORB_create(nfeatures)
    feats = []
    for i, t in enumerate(tiles):
        f = cv2.detail.computeImageFeatures2(orb, t)
        f.img_idx = i
        feats.append(f)
    return feats


def tile_json(f):
    pts = np.array([kp.pt for kp in f.keypoints], np.float32).reshape(-1, 2)
    desc = f.descriptors.get() if isinstance(f.descriptors, cv2.UMat) else np.asarray(f.descriptors)
    desc = np.asarray(desc, np.uint8).reshape(len(f.keypoints), -1) if len(f.keypoints) else np.zeros((0, 32), np.uint8)
    return {"width": int(f.img_size[0]), "height": int(f.img_size[1]), "count": len(f.keypoints),
            "pts": b64(pts), "desc": b64(desc), "descCols": int(desc.shape[1]) if desc.size else 32}


def kp_pts(f):
    return np.array([kp.pt for kp in f.keypoints], np.float32).reshape(-1, 2)


def desc_of(f):
    d = f.descriptors.get() if isinstance(f.descriptors, cv2.UMat) else np.asarray(f.descriptors)
    return np.asarray(d, np.uint8)


# ---------------------------------------------------------------- serialization

def mi_json(m):
    matches = list(m.matches)
    q = np.array([d.queryIdx for d in matches], np.int32)
    t = np.array([d.trainIdx for d in matches], np.int32)
    dist = np.array([d.distance for d in matches], np.float32)
    mask = np.asarray(m.inliers_mask, np.uint8).reshape(-1) if m.inliers_mask is not None else np.zeros(0, np.uint8)
    out = {"src": int(m.src_img_idx), "dst": int(m.dst_img_idx), "count": len(matches),
           "query": b64(q), "train": b64(t), "dist": b64(dist), "maskLength": int(mask.size), "mask": b64(mask),
           "numInliers": int(m.num_inliers), "confidence": float(m.confidence)}
    h = m.H
    if h is not None and np.asarray(h).size > 0:
        out["H"] = b64(np.asarray(h, np.float64))
    return out


class RefMI:
    def __init__(self):
        self.src_img_idx = -1
        self.dst_img_idx = -1
        self.matches = []
        self.inliers_mask = np.zeros(0, np.uint8)
        self.num_inliers = 0
        self.H = None
        self.confidence = 0.0


def cam_json(c):
    r = np.asarray(c.R)
    return {"focal": float(c.focal), "aspect": float(c.aspect), "ppx": float(c.ppx), "ppy": float(c.ppy),
            "R": b64(r.astype(np.float64)), "rType": "f32" if r.dtype == np.float32 else "f64"}


# ---------------------------------------------------------------- Python reference of matchers.cpp with BFMatcher

def ref_cpu_match(f1, f2, conf):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    d1, d2 = desc_of(f1), desc_of(f2)
    ratio = F32(F32(1.0) - F32(conf))
    out = []
    seen = set()
    for pair in bf.knnMatch(d1, d2, k=2):
        if len(pair) < 2:
            continue
        m0, m1 = pair
        if F32(m0.distance) < F32(ratio * F32(m1.distance)):
            out.append(cv2.DMatch(m0.queryIdx, m0.trainIdx, m0.distance))
            seen.add((m0.queryIdx, m0.trainIdx))
    for pair in bf.knnMatch(d2, d1, k=2):
        if len(pair) < 2:
            continue
        m0, m1 = pair
        if F32(m0.distance) < F32(ratio * F32(m1.distance)) and (m0.trainIdx, m0.queryIdx) not in seen:
            out.append(cv2.DMatch(m0.trainIdx, m0.queryIdx, m0.distance))
    return out


def centered(f, idx):
    p = kp_pts(f)[idx]
    w, h = f.img_size
    return np.stack([p[:, 0] - F32(F32(w) * F32(0.5)), p[:, 1] - F32(F32(h) * F32(0.5))], axis=1).astype(np.float32)


def ref_best_match(f1, f2, mi, conf, t1, t2, conf_thresh):
    mi.matches = ref_cpu_match(f1, f2, conf)
    n = len(mi.matches)
    if n < t1:
        return
    q = np.array([m.queryIdx for m in mi.matches])
    t = np.array([m.trainIdx for m in mi.matches])
    src, dst = centered(f1, q), centered(f2, t)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    mi.H = h
    mi.inliers_mask = np.zeros(0, np.uint8) if mask is None else mask.reshape(-1).astype(np.uint8)
    if h is None or abs(cv2.determinant(h)) < np.finfo(np.float64).eps:
        return
    mi.num_inliers = int(np.count_nonzero(mi.inliers_mask))
    c = mi.num_inliers / (8 + 0.3 * n)
    mi.confidence = 0.0 if c > conf_thresh else c
    if mi.num_inliers < t2:
        return
    sel = mi.inliers_mask != 0
    h2, _ = cv2.findHomography(src[sel], dst[sel], cv2.RANSAC)
    mi.H = h2


def ref_affine_match(f1, f2, mi, conf, t1, full):
    mi.matches = ref_cpu_match(f1, f2, conf)
    n = len(mi.matches)
    if n < t1:
        return
    q = np.array([m.queryIdx for m in mi.matches])
    t = np.array([m.trainIdx for m in mi.matches])
    src, dst = kp_pts(f1)[q], kp_pts(f2)[t]
    if full:
        h, mask = cv2.estimateAffine2D(src, dst)
    else:
        h, mask = cv2.estimateAffinePartial2D(src, dst)
    mi.inliers_mask = np.zeros(0, np.uint8) if mask is None else mask.reshape(-1).astype(np.uint8)
    if h is None:
        mi.H = None
        mi.confidence = 0.0
        mi.num_inliers = 0
        return
    mi.num_inliers = int(np.count_nonzero(mi.inliers_mask))
    mi.confidence = mi.num_inliers / (8 + 0.3 * n)
    mi.H = np.vstack([h, [0.0, 0.0, 1.0]])


def ref_pairwise(feats, spec, mask):
    n = len(feats)
    width = spec.get("rangeWidth", n + 1)
    pw = [RefMI() for _ in range(n * n)]
    for i in range(n - 1):
        for j in range(i + 1, min(n, i + 1 + width)):
            if len(feats[i].keypoints) == 0 or len(feats[j].keypoints) == 0:
                continue
            if mask is not None and mask[i, j] == 0:
                continue
            mi = pw[i * n + j]
            if spec["kind"] == "affine":
                ref_affine_match(feats[i], feats[j], mi, spec["matchConf"], spec["thresh1"], spec["fullAffine"])
            else:
                ref_best_match(feats[i], feats[j], mi, spec["matchConf"], spec["thresh1"], spec["thresh2"],
                               spec["confThresh"])
            mi.src_img_idx, mi.dst_img_idx = i, j
            d = RefMI()
            d.src_img_idx, d.dst_img_idx = j, i
            d.matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in mi.matches]
            d.inliers_mask = mi.inliers_mask.copy()
            d.num_inliers = mi.num_inliers
            d.confidence = mi.confidence
            d.H = None if mi.H is None else cv2.invert(mi.H, flags=cv2.DECOMP_LU)[1]
            pw[j * n + i] = d
    return pw


def cv2_matcher(spec):
    k = spec["kind"]
    if k == "best":
        return cv2.detail_BestOf2NearestMatcher(False, spec["matchConf"], spec["thresh1"], spec["thresh2"],
                                                spec["confThresh"])
    if k == "range":
        return cv2.detail_BestOf2NearestRangeMatcher(spec["rangeWidth"], False, spec["matchConf"], spec["thresh1"],
                                                     spec["thresh2"])
    return cv2.detail_AffineBestOf2NearestMatcher(spec["fullAffine"], False, spec["matchConf"], spec["thresh1"])


def pw_json(pw):
    return [mi_json(m) for m in pw]


# ---------------------------------------------------------------- battery

class Gen:
    def __init__(self):
        self.tile_sets = {}
        self.feats = {}
        self.matcher_cases = []
        self.cv2_pairwise = {}
        self.estimator_cases = []
        self.focal_cases = []
        self.component_cases = []
        self.calib_cases = []

    def add_set(self, name, tiles, nfeatures=500):
        feats = features_of(tiles, nfeatures)
        self.feats[name] = feats
        self.tile_sets[name] = [tile_json(f) for f in feats]

    def add_matcher(self, name, set_name, spec, mask=None):
        feats = self.feats[set_name]
        m = cv2_matcher(spec)
        args = (feats,) if mask is None else (feats, mask)
        pw = m.apply2(*args)
        out1 = pw_json(pw)
        out2 = pw_json(cv2_matcher(spec).apply2(*args))
        ref1 = pw_json(ref_pairwise(feats, spec, mask))
        ref2 = pw_json(ref_pairwise(feats, spec, mask))
        assert ref1 == ref2, name
        agrees = all(a["query"] == b["query"] and a["train"] == b["train"] for a, b in zip(out1, ref1))
        case = {"name": name, "set": set_name, "spec": spec, "cv2": out1, "reference": ref1,
                "flannAgrees": agrees, "flannRepeatable": out1 == out2, "suspect": name in SUSPECT_CASES}
        if mask is not None:
            case["mask"] = b64(mask.astype(np.uint8))
        self.matcher_cases.append(case)
        self.cv2_pairwise[name] = pw

    def add_estimator(self, name, matcher_case, estimator, adjusters, waves):
        set_name = next(c["set"] for c in self.matcher_cases if c["name"] == matcher_case)
        feats = self.feats[set_name]
        pw = self.cv2_pairwise[matcher_case]

        def run_est():
            e = cv2.detail_HomographyBasedEstimator() if estimator == "homography" else cv2.detail_AffineBasedEstimator()
            ok, cams = e.apply(feats, pw, None)
            return ok, cams

        ok, cams = run_est()
        ok2, cams2 = run_est()
        assert ok == ok2 and [cam_json(c) for c in cams] == [cam_json(c) for c in cams2], name
        case = {"name": name, "matcherCase": matcher_case, "estimator": estimator, "ok": bool(ok),
                "cameras": [cam_json(c) for c in cams], "adjusters": [], "suspect": name in SUSPECT_CASES}
        for adj in adjusters:
            def run_ba():
                cin = []
                for c in cams:
                    d = cv2.detail.CameraParams()
                    d.focal, d.aspect, d.ppx, d.ppy = c.focal, c.aspect, c.ppx, c.ppy
                    d.R = np.asarray(c.R).astype(np.float32)
                    d.t = np.asarray(c.t)
                    cin.append(d)
                b = {"ray": cv2.detail_BundleAdjusterRay, "reproj": cv2.detail_BundleAdjusterReproj,
                     "affine": cv2.detail_BundleAdjusterAffine, "partial": cv2.detail_BundleAdjusterAffinePartial,
                     "no": cv2.detail_NoBundleAdjuster}[adj["kind"]]()
                b.setConfThresh(adj["confThresh"])
                if "refinementMask" in adj:
                    b.setRefinementMask(np.array(adj["refinementMask"], np.uint8).reshape(3, 3))
                okb, cout = b.apply(feats, pw, cin)
                return okb, cout

            okb, cout = run_ba()
            okb2, cout2 = run_ba()
            assert okb == okb2 and [cam_json(c) for c in cout] == [cam_json(c) for c in cout2], name
            entry = dict(adj)
            per_cam = {"ray": 4, "reproj": 7, "affine": 6, "partial": 4, "no": 0}[adj["kind"]]
            entry["lapackSolve"] = per_cam * len(feats) >= 25
            entry["ok"] = bool(okb)
            entry["cameras"] = [cam_json(c) for c in cout]
            entry["waves"] = []
            if okb:
                for kind in waves:
                    rm = [np.asarray(c.R).astype(np.float32).copy() for c in cout]
                    w1 = cv2.detail.waveCorrect([r.copy() for r in rm], kind)
                    w2 = cv2.detail.waveCorrect([r.copy() for r in rm], kind)
                    assert all(np.array_equal(a, b) for a, b in zip(w1, w2)), name
                    entry["waves"].append({"kind": int(kind),
                                           "R": [b64(np.asarray(r).astype(np.float64)) for r in w1]})
            case["adjusters"].append(entry)
        self.estimator_cases.append(case)

    def add_focal(self, name, h):
        f0, f1, ok0, ok1 = ref_focals(h)
        self.focal_cases.append({"name": name, "H": b64(np.asarray(h, np.float64)), "f": b64(np.array([f0, f1], np.float64)),
                                 "f0Ok": ok0, "f1Ok": ok1})

    def add_component(self, name, set_name, matcher_case, conf, paths):
        feats = self.feats[set_name]
        pw = self.cv2_pairwise[matcher_case]
        idx = cv2.detail.leaveBiggestComponent(list(feats), list(pw), conf)
        idx2 = cv2.detail.leaveBiggestComponent(list(feats), list(pw), conf)
        assert np.array_equal(idx, idx2)
        graph = cv2.detail.matchesGraphAsString(paths, list(pw), conf)
        self.component_cases.append({"name": name, "matcherCase": matcher_case, "conf": conf,
                                     "indices": [int(v) for v in np.asarray(idx).reshape(-1)],
                                     "paths": paths, "graph": graph})

    def add_calib(self, name, hs):
        ok, k = cv2.detail.calibrateRotatingCamera([np.asarray(h, np.float64) for h in hs])
        case = {"name": name, "Hs": [b64(np.asarray(h, np.float64)) for h in hs], "ok": bool(ok)}
        if ok:
            case["K"] = b64(np.asarray(k, np.float64))
        self.calib_cases.append(case)


def ref_focals(h):
    with np.errstate(all="ignore"):
        return _ref_focals(h)


def _ref_focals(h):
    # autocalib.cpp focalsFromHomography; cv2 does not return the output arguments.
    h = np.asarray(h, np.float64).reshape(-1)
    h = [np.float64(v) for v in h]
    f0 = f1 = 0.0
    f1_ok = True
    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2
    if v1 < v2:
        v1, v2 = v2, v1
        d1, d2 = d2, d1
    if v1 > 0 and v2 > 0:
        f1 = float(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
    elif v1 > 0:
        f1 = float(np.sqrt(v1))
    else:
        f1_ok = False
    f0_ok = True
    d1 = h[0] * h[3] + h[1] * h[4]
    d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4]
    v1 = -h[2] * h[5] / d1
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2
    if v1 < v2:
        v1, v2 = v2, v1
        d1, d2 = d2, d1
    if v1 > 0 and v2 > 0:
        f0 = float(np.sqrt(v1 if abs(d1) > abs(d2) else v2))
    elif v1 > 0:
        f0 = float(np.sqrt(v1))
    else:
        f0_ok = False
    return f0, f1, f0_ok, f1_ok


def main():
    scene = make_scene()
    g = Gen()
    g.add_set("affine", affine_tiles(scene))
    g.add_set("homography", homography_tiles(scene))
    yaw = [-0.3, -0.15, 0.0, 0.15, 0.3]
    g.add_set("yaw", rotation_tiles(scene, yaw, "yaw"))
    g.add_set("pitch", rotation_tiles(scene, [-0.2, -0.1, 0.0, 0.1, 0.2], "pitch"))
    other = make_scene(seed=99)
    g.add_set("yaw3", rotation_tiles(scene, [-0.15, 0.0, 0.15], "yaw"))
    g.add_set("pitch3", rotation_tiles(scene, [-0.1, 0.0, 0.1], "pitch"))
    g.add_set("affine4", affine_tiles(scene, count=4))
    g.add_set("mixed", affine_tiles(scene, count=4) + [cv2.resize(other[:320, :400], (400, 320))])

    best = {"kind": "best", "matchConf": 0.3, "thresh1": 6, "thresh2": 6, "confThresh": 3.0}
    g.add_matcher("best_default_affine", "affine", best)
    g.add_matcher("best_default_homography", "homography", best)
    g.add_matcher("best_default_yaw", "yaw", best)
    g.add_matcher("best_default_pitch", "pitch", best)
    g.add_matcher("best_conf065_yaw", "yaw", dict(best, matchConf=0.65))
    g.add_matcher("best_thresh_yaw", "yaw", dict(best, thresh1=20, thresh2=60, confThresh=1.5))
    g.add_matcher("best_conf01_homography", "homography", dict(best, matchConf=0.1))
    rng_spec = {"kind": "range", "matchConf": 0.3, "thresh1": 6, "thresh2": 6, "confThresh": 3.0}
    g.add_matcher("range2_affine", "affine", dict(rng_spec, rangeWidth=2))
    g.add_matcher("range3_yaw", "yaw", dict(rng_spec, rangeWidth=3))
    g.add_matcher("affine_partial", "affine", {"kind": "affine", "matchConf": 0.3, "thresh1": 6, "thresh2": 6,
                                               "fullAffine": False})
    g.add_matcher("affine_full", "affine", {"kind": "affine", "matchConf": 0.3, "thresh1": 6, "thresh2": 6,
                                            "fullAffine": True})
    g.add_matcher("affine_partial_mixed", "mixed", {"kind": "affine", "matchConf": 0.3, "thresh1": 6,
                                                    "thresh2": 6, "fullAffine": False})
    g.add_matcher("best_mixed", "mixed", best)
    g.add_matcher("best_default_yaw3", "yaw3", best)
    g.add_matcher("best_default_pitch3", "pitch3", best)
    g.add_matcher("affine_full4", "affine4", {"kind": "affine", "matchConf": 0.3, "thresh1": 6, "thresh2": 6,
                                              "fullAffine": True})
    mask = np.ones((5, 5), np.uint8)
    mask[0, 2] = mask[1, 3] = mask[0, 4] = 0
    g.add_matcher("best_mask_yaw", "yaw", best, mask)

    hz, vt, au = cv2.detail.WAVE_CORRECT_HORIZ, cv2.detail.WAVE_CORRECT_VERT, cv2.detail.WAVE_CORRECT_AUTO
    # Adjusters with 25 or more parameters solve through the cv2 LAPACK SVD (see the notes);
    # the battery keeps them few, and the generator marks them with "lapackSolve".
    rot_adj = [{"kind": "ray", "confThresh": 1.0}, {"kind": "ray", "confThresh": 2.0},
               {"kind": "no", "confThresh": 1.0}]
    reproj_adj = [{"kind": "reproj", "confThresh": 1.0},
                  {"kind": "reproj", "confThresh": 1.0, "refinementMask": [1, 0, 0, 0, 0, 0, 0, 0, 0]},
                  {"kind": "reproj", "confThresh": 1.0, "refinementMask": [1, 0, 1, 0, 1, 1, 0, 0, 0]},
                  {"kind": "ray", "confThresh": 1.0}]
    g.add_estimator("homography_yaw", "best_default_yaw", "homography", rot_adj, [hz, vt, au])
    g.add_estimator("homography_pitch", "best_default_pitch", "homography", rot_adj[:1], [hz, vt, au])
    g.add_estimator("homography_homography", "best_default_homography", "homography", rot_adj[:1], [hz, au])
    g.add_estimator("homography_mask_yaw", "best_mask_yaw", "homography", rot_adj[:1], [hz])
    g.add_estimator("homography_yaw3", "best_default_yaw3", "homography", reproj_adj, [hz, au])
    g.add_estimator("homography_pitch3", "best_default_pitch3", "homography", reproj_adj[:1], [vt, au])
    aff_adj = [{"kind": "partial", "confThresh": 1.0}, {"kind": "partial", "confThresh": 2.0},
               {"kind": "no", "confThresh": 1.0}, {"kind": "affine", "confThresh": 1.0}]
    g.add_estimator("affine_partial", "affine_partial", "affine", aff_adj, [])
    g.add_estimator("affine_full", "affine_full", "affine", aff_adj[:1] + aff_adj[3:], [])
    g.add_estimator("affine4_full", "affine_full4", "affine", aff_adj[3:] + aff_adj[:1], [])

    for c in g.matcher_cases:
        for k, mi in enumerate(c["cv2"]):
            if "H" in mi and k % 3 == 1:
                h = np.frombuffer(base64.b64decode(mi["H"]), np.float64).reshape(3, 3)
                g.add_focal(f"focal_{c['name']}_{k}", h)
    g.add_focal("focal_identity", np.eye(3))

    g.add_component("component_mixed_best", "mixed", "best_mixed", 1.0, ["a/t0.png", "a\\t1.png", "t2", "t3", "x/odd.png"])
    g.add_component("component_mixed_affine", "mixed", "affine_partial_mixed", 1.0, ["t0", "t1", "t2", "t3", "odd"])
    g.add_component("component_yaw_high", "yaw", "best_default_yaw", 2.0, ["y0", "y1", "y2", "y3", "y4"])

    hs = [np.frombuffer(base64.b64decode(mi["H"]), np.float64).reshape(3, 3)
          for mi in next(c for c in g.matcher_cases if c["name"] == "best_default_yaw")["cv2"] if "H" in mi]
    g.add_calib("calib_yaw_all", hs)
    g.add_calib("calib_yaw_two", hs[:2])

    out = {"generator": "generate_stitch_reg_golden.py", "cv2Version": cv2.__version__,
           "tileSets": g.tile_sets, "matcherCases": g.matcher_cases, "estimatorCases": g.estimator_cases,
           "focalCases": g.focal_cases, "componentCases": g.component_cases, "calibCases": g.calib_cases}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "stitch_reg_golden.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    agree = sum(c["flannAgrees"] for c in g.matcher_cases)
    print(f"wrote {path}: {len(g.matcher_cases)} matcher cases ({agree} FLANN agrees), "
          f"{len(g.estimator_cases)} estimator cases, {len(g.focal_cases)} focal, "
          f"{len(g.component_cases)} component, {len(g.calib_cases)} calib; "
          f"{os.path.getsize(path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

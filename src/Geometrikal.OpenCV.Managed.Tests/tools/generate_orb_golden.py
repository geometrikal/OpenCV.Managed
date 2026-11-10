#!/usr/bin/env python3
"""Golden-data generator for the managed FAST and ORB port.

Runs cv2 5.0.0 ``FastFeatureDetector`` and ``ORB`` (detect, compute, detectAndCompute) over a
seeded battery of synthetic 8-bit images and writes ``TestData/orb_golden.json``.
``Features/FastGoldenTests.cs`` and ``Features/OrbGoldenTests.cs`` replay the cases.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>``. numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set). FAST then uses the 128-bit SIMD path
   and fastAtan2 uses the SSE baseline without FMA.
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_orb_golden.py

Battery notes:
 * Images are not stored. Each case stores an image recipe (``image``) that the C# tests
   rebuild with integer arithmetic only: hashed noise smoothed by an integer box mean and
   stretched, filled rectangles and circles, checkerboards and hashed jitter. Masks are also
   recipes.
 * FAST keypoints are stored as int32 triples (x, y, response); cv2 order is row-major and is
   compared exactly. Lists of more than 3000 keypoints are stored as SHA-256 (``keypointsSha``).
 * ORB keypoints are stored as float32 rows (x, y, size, angle, response, octave). OpenCV orders
   the keypoints of a level with std::nth_element, so the tests compare the keypoint sets sorted
   by (octave, y, x). Descriptors are stored in the cv2 keypoint order and are compared per
   keypoint. ``compute`` cases pass the stored keypoints in a fixed order and compare the
   output order exactly.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os
import platform
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_orb_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {}
FAST_TYPES = {"Type5_8": cv2.FastFeatureDetector_TYPE_5_8, "Type7_12": cv2.FastFeatureDetector_TYPE_7_12,
              "Type9_16": cv2.FastFeatureDetector_TYPE_9_16}
SCORES = {"Harris": cv2.ORB_HARRIS_SCORE, "Fast": cv2.ORB_FAST_SCORE}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def hash_u32(x, y):
    h = (np.asarray(x, dtype=np.uint32) * np.uint32(73856093)) ^ (np.asarray(y, dtype=np.uint32) * np.uint32(19349663))
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h


def build_image(spec):
    """Rebuilds an image recipe. The C# test helper OrbImages.Build implements the same steps."""
    w, h, seed = spec["width"], spec["height"], spec["seed"]
    kind = spec["kind"]
    if kind == "noise":
        r, gain = spec["radius"], spec["gain"]
        ys, xs = np.mgrid[0:h + 2 * r, 0:w + 2 * r].astype(np.uint32)
        base = (hash_u32(xs + np.uint32(131 * seed), ys) & np.uint32(255)).astype(np.int64)
        k = 2 * r + 1
        c = np.zeros((h + 2 * r + 1, w + 2 * r + 1), np.int64)
        c[1:, 1:] = base.cumsum(0).cumsum(1)
        s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
        avg = s // (k * k)
        img = np.clip(128 + (avg - 128) * gain, 0, 255)
    elif kind == "checker":
        sq = spec["square"]
        ys, xs = np.mgrid[0:h, 0:w]
        img = np.where(((xs // sq) + (ys // sq)) % 2 == 0, spec["low"], spec["high"]).astype(np.int64)
    else:
        img = np.full((h, w), spec["background"], np.int64)
    ys, xs = np.mgrid[0:h, 0:w]
    for s in spec.get("shapes", []):
        if s[0] == 0:
            x0, y0, x1, y1, v = s[1:]
            img[(xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)] = v
        else:
            cx, cy, rr, v = s[1:]
            img[(xs - cx) ** 2 + (ys - cy) ** 2 <= rr * rr] = v
    if spec.get("jitter", 0) > 0:
        j = spec["jitter"]
        hv = (hash_u32(xs.astype(np.uint32) + np.uint32(977), ys.astype(np.uint32) + np.uint32(7 * seed)) % np.uint32(2 * j + 1)).astype(np.int64)
        img = np.clip(img + hv - j, 0, 255)
    return np.ascontiguousarray(img.astype(np.uint8))


def build_mask(spec, w, h):
    """Mask recipe: rectangles and circles set to the given value on a zero background."""
    if spec is None:
        return None
    m = np.zeros((h, w), np.int64)
    ys, xs = np.mgrid[0:h, 0:w]
    for s in spec:
        if s[0] == 0:
            x0, y0, x1, y1, v = s[1:]
            m[(xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)] = v
        else:
            cx, cy, rr, v = s[1:]
            m[(xs - cx) ** 2 + (ys - cy) ** 2 <= rr * rr] = v
    return m.astype(np.uint8)


def random_shapes(rng, w, h, n):
    shapes = []
    for _ in range(n):
        v = int(rng.integers(0, 256))
        if rng.random() < 0.5:
            x0, y0 = int(rng.integers(0, w - 4)), int(rng.integers(0, h - 4))
            shapes.append([0, x0, y0, x0 + int(rng.integers(4, max(5, w // 3))), y0 + int(rng.integers(4, max(5, h // 3))), v])
        else:
            shapes.append([1, int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(3, max(4, min(w, h) // 5))), v])
    return shapes


def images(rng):
    out = {}
    for (w, h) in ((64, 64), (160, 120), (321, 243), (640, 480)):
        out[f"noise_{w}x{h}"] = {"kind": "noise", "width": w, "height": h, "seed": w + h, "radius": 2, "gain": 4, "jitter": 0}
    out["noise_r1_200x150"] = {"kind": "noise", "width": 200, "height": 150, "seed": 5, "radius": 1, "gain": 2, "jitter": 0}
    out["noise_r3_256x256"] = {"kind": "noise", "width": 256, "height": 256, "seed": 9, "radius": 3, "gain": 6, "jitter": 0}
    for (w, h) in ((96, 80), (256, 192), (480, 360)):
        out[f"shapes_{w}x{h}"] = {"kind": "flat", "width": w, "height": h, "seed": w, "background": 90,
                                  "shapes": random_shapes(rng, w, h, 25), "jitter": 3}
    out["checker_200x150"] = {"kind": "checker", "width": 200, "height": 150, "seed": 3, "square": 13, "low": 40, "high": 210, "jitter": 5}
    out["checker_128x128"] = {"kind": "checker", "width": 128, "height": 128, "seed": 4, "square": 8, "low": 100, "high": 140, "jitter": 0}
    out["mixed_400x300"] = {"kind": "noise", "width": 400, "height": 300, "seed": 77, "radius": 2, "gain": 3,
                            "shapes": random_shapes(rng, 400, 300, 30), "jitter": 2}
    out["tiny_7x7"] = {"kind": "noise", "width": 7, "height": 7, "seed": 1, "radius": 0, "gain": 1, "jitter": 0}
    out["narrow_40x9"] = {"kind": "noise", "width": 40, "height": 9, "seed": 2, "radius": 0, "gain": 1, "jitter": 0}
    out["narrow_9x40"] = {"kind": "noise", "width": 9, "height": 40, "seed": 3, "radius": 0, "gain": 1, "jitter": 0}
    out["odd_37x29"] = {"kind": "noise", "width": 37, "height": 29, "seed": 4, "radius": 1, "gain": 3, "jitter": 0}
    return out


def kp_rows(kps):
    return np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, float(k.octave)] for k in kps], np.float32).reshape(-1, 6)


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()

    def add(self, case):
        assert case["name"] not in self.names, case["name"]
        self.names.add(case["name"])
        case["suspect"] = case["name"] in SUSPECT_CASES
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[case["name"]]
        self.cases.append(case)


def fast_cases(g, imgs):
    plan = []
    for name in ("noise_64x64", "noise_160x120", "noise_321x243", "noise_r1_200x150", "shapes_96x80",
                 "shapes_256x192", "checker_200x150", "checker_128x128", "tiny_7x7", "narrow_40x9", "narrow_9x40", "odd_37x29"):
        for t in FAST_TYPES:
            for thr in (10, 25):
                for nms in (True, False):
                    plan.append((name, t, thr, nms, None))
    for t in FAST_TYPES:
        plan.append(("noise_640x480", t, 20, True, None))
        plan.append(("mixed_400x300", t, 15, True, None))
        plan.append(("noise_r3_256x256", t, 0, True, None))
        plan.append(("noise_r3_256x256", t, 1, False, None))
        plan.append(("checker_128x128", t, 255, True, None))
        plan.append(("noise_160x120", t, 12, True, [[0, 20, 10, 120, 90, 1], [1, 40, 60, 25, 0]]))
        plan.append(("shapes_256x192", t, 8, False, [[1, 128, 96, 80, 255]]))
    for (name, t, thr, nms, mspec) in plan:
        spec = imgs[name]
        img = build_image(spec)
        mask = build_mask(mspec, spec["width"], spec["height"])
        det = cv2.FastFeatureDetector_create(thr, nms, FAST_TYPES[t])
        kps = det.detect(img, mask)
        rows = []
        for k in kps:
            assert k.size == 7 and k.angle == -1 and k.octave == 0 and k.class_id == -1
            assert k.response == int(k.response)
            rows.append([int(k.pt[0]), int(k.pt[1]), int(k.response)])
        if not nms:
            assert all(r[2] == 0 for r in rows)
        case = {"name": f"fast_{name}_{t}_t{thr}_{'nms' if nms else 'all'}{'_mask' if mspec else ''}",
                "op": "fast", "image": spec, "mask": mspec,
                "params": {"threshold": thr, "nonmax": nms, "type": t},
                "count": len(rows)}
        arr = np.array(rows, np.int32).reshape(-1, 3)
        if len(rows) > 3000:
            case["keypointsSha"] = hashlib.sha256(arr.tobytes()).hexdigest()
        else:
            case["keypoints"] = b64(arr)
        g.add(case)


ORB_DEFAULT = {"nfeatures": 500, "scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 31, "firstLevel": 0,
               "wtaK": 2, "scoreType": "Harris", "patchSize": 31, "fastThreshold": 20}


def make_orb(p):
    return cv2.ORB_create(nfeatures=p["nfeatures"], scaleFactor=p["scaleFactor"], nlevels=p["nlevels"],
                          edgeThreshold=p["edgeThreshold"], firstLevel=p["firstLevel"], WTA_K=p["wtaK"],
                          scoreType=SCORES[p["scoreType"]], patchSize=p["patchSize"], fastThreshold=p["fastThreshold"])


def orb_cases(g, imgs):
    variations = [
        ("default", {}),
        ("n100", {"nfeatures": 100}),
        ("n2000", {"nfeatures": 2000}),
        ("sf1_5", {"scaleFactor": 1.5}),
        ("sf2", {"scaleFactor": 2.0, "nlevels": 4}),
        ("lv1", {"nlevels": 1}),
        ("lv3", {"nlevels": 3}),
        ("edge10", {"edgeThreshold": 10}),
        ("edge3_patch15", {"edgeThreshold": 3, "patchSize": 15}),
        ("first1", {"firstLevel": 1}),
        ("first2_lv4", {"firstLevel": 2, "nlevels": 4}),
        ("wta3", {"wtaK": 3}),
        ("wta4", {"wtaK": 4}),
        ("fastscore", {"scoreType": "Fast"}),
        ("fastscore_n50", {"scoreType": "Fast", "nfeatures": 50}),
        ("patch21", {"patchSize": 21, "edgeThreshold": 21}),
        ("patch45_wta4", {"patchSize": 45, "edgeThreshold": 45, "wtaK": 4}),
        ("fth5", {"fastThreshold": 5}),
        ("fth40", {"fastThreshold": 40}),
    ]
    plan = []
    for (vname, v) in variations:
        for iname in ("noise_321x243", "shapes_256x192"):
            plan.append((iname, vname, v, None))
    for iname in ("noise_640x480", "mixed_400x300", "checker_200x150", "noise_r3_256x256", "shapes_480x360", "noise_160x120",
                  "shapes_96x80", "noise_64x64", "checker_128x128"):
        plan.append((iname, "default", {}, None))
    plan.append(("noise_640x480", "wta3_fastscore", {"wtaK": 3, "scoreType": "Fast", "nfeatures": 1000}, None))
    plan.append(("mixed_400x300", "mask_rect", {}, [[0, 50, 40, 330, 260, 1]]))
    plan.append(("mixed_400x300", "mask_circles", {"nfeatures": 800}, [[1, 200, 150, 120, 255], [1, 60, 60, 50, 7], [0, 180, 140, 220, 160, 0]]))
    plan.append(("noise_321x243", "mask_first1", {"firstLevel": 1, "nlevels": 5}, [[0, 30, 30, 290, 200, 200]]))
    plan.append(("shapes_256x192", "mask_fastscore", {"scoreType": "Fast"}, [[1, 128, 96, 90, 255]]))

    for (iname, vname, v, mspec) in plan:
        p = dict(ORB_DEFAULT, **v)
        spec = imgs[iname]
        img = build_image(spec)
        mask = build_mask(mspec, spec["width"], spec["height"])
        orb = make_orb(p)
        kps, desc = orb.detectAndCompute(img, mask)
        kps_d = orb.detect(img, mask)
        rows = kp_rows(kps)
        rows_d = kp_rows(kps_d)
        assert sorted(map(tuple, rows.tolist())) == sorted(map(tuple, rows_d.tolist())), (iname, vname)
        if desc is None:
            desc = np.zeros((0, 32), np.uint8)
        assert desc.shape == (len(kps), 32)
        pstore = dict(p, scaleFactor=float(np.float32(p["scaleFactor"])))
        g.add({"name": f"orb_{iname}_{vname}", "op": "orb", "image": spec, "mask": mspec, "params": pstore,
               "count": len(kps), "keypoints": b64(rows), "descriptors": b64(desc)})

    # compute with provided keypoints: cv2 order is kept unless octaves are unsorted.
    for (iname, vname, v) in (("noise_321x243", "default", {}), ("shapes_256x192", "wta4", {"wtaK": 4}),
                              ("mixed_400x300", "edge5_first1", {"edgeThreshold": 5, "firstLevel": 1}),
                              ("noise_640x480", "patch21", {"patchSize": 21, "edgeThreshold": 21})):
        p = dict(ORB_DEFAULT, **v)
        spec = imgs[iname]
        img = build_image(spec)
        orb = make_orb(p)
        kps = list(orb.detect(img, None))
        kps.sort(key=lambda k: (k.pt[1], k.pt[0]))
        for shuffled in (False, True):
            inp = list(kps)
            if not shuffled:
                inp.sort(key=lambda k: k.octave)
            out_kps, desc = orb.compute(img, inp)
            pstore = dict(p, scaleFactor=float(np.float32(p["scaleFactor"])))
            g.add({"name": f"orbcompute_{iname}_{vname}_{'unsorted' if shuffled else 'sorted'}", "op": "orbCompute",
                   "image": spec, "mask": None, "params": pstore,
                   "inputCount": len(inp), "inputKeypoints": b64(kp_rows(inp)),
                   "count": len(out_kps), "keypoints": b64(kp_rows(out_kps)),
                   "descriptors": b64(desc if desc is not None else np.zeros((0, 32), np.uint8))})

    # compute with hand-made keypoints: angles and sub-pixel positions near rounding ties.
    spec = imgs["noise_321x243"]
    img = build_image(spec)
    rng = np.random.default_rng(321)
    inp = []
    for i in range(300):
        octave = int(rng.integers(0, 4))
        x = float(np.float32(rng.uniform(40, 280)))
        y = float(np.float32(rng.uniform(40, 200)))
        if i % 3 == 0:
            x = float(int(x)) + 0.5
        ang = float(np.float32(rng.uniform(0, 360))) if i % 5 else float(90 * (i % 4))
        inp.append(cv2.KeyPoint(x, y, 31.0, ang, 0.0, octave))
    for wta in (2, 3, 4):
        p = dict(ORB_DEFAULT, wtaK=wta, edgeThreshold=31)
        orb = make_orb(p)
        out_kps, desc = orb.compute(img, inp)
        pstore = dict(p, scaleFactor=float(np.float32(p["scaleFactor"])))
        g.add({"name": f"orbcompute_manual_wta{wta}", "op": "orbCompute", "image": spec, "mask": None, "params": pstore,
               "inputCount": len(inp), "inputKeypoints": b64(kp_rows(inp)),
               "count": len(out_kps), "keypoints": b64(kp_rows(out_kps)), "descriptors": b64(desc)})


def main():
    rng = np.random.default_rng(20261016)
    imgs = images(rng)
    g = Gen()
    fast_cases(g, imgs)
    orb_cases(g, imgs)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    doc = {"generator": {"cv2": cv2.__version__, "numpy": np.__version__, "machine": platform.processor()},
           "cases": g.cases}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "orb_golden.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()

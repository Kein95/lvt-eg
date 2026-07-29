"""Dataset-parser smoke tests - verify modules import and signatures hold.

These tests do NOT require dataset files on disk. They exercise the parsers
on temp directories with minimal fixtures so CI runs offline.
"""

from pathlib import Path


def test_rodosol_signature_and_empty_dir(tmp_path: Path):
    from lvt_eg.data.rodosol import parse_rodosol_merged, parse_rodosol_split
    (tmp_path / "split.txt").write_text("")
    assert parse_rodosol_split(tmp_path, "training") == []
    assert parse_rodosol_merged(tmp_path, ["training", "validation"]) == []


def test_lp2025_signature_missing_dirs(tmp_path: Path):
    from lvt_eg.data.lp2025 import parse_lp2025_split
    assert parse_lp2025_split(tmp_path, "train") == []


def test_ukrainian_flat_empty(tmp_path: Path):
    from lvt_eg.data.ukrainian import parse_ukrainian_dataset
    train, val, test = parse_ukrainian_dataset(tmp_path, "ABC0123456789", seed=42)
    assert train == [] and val == [] and test == []


def test_ccpd_signature(tmp_path: Path):
    from lvt_eg.data.ccpd import CCPD_SUBSETS, parse_ccpd_subsets
    out = parse_ccpd_subsets(tmp_path)
    assert set(out) == set(CCPD_SUBSETS)
    assert all(v == [] for v in out.values())


def test_lplc_lp_filename_regex():
    from lvt_eg.data.lplc import parse_lp_filename
    p = "LPRD/all_lps/perfect/abc123-def.jpg_0_AAA1A11.jpg"
    parsed = parse_lp_filename(p)
    assert parsed is not None
    image_id, lp_idx, ocr = parsed
    assert image_id == "abc123-def"
    assert lp_idx == 0
    assert ocr == "AAA1A11"


def test_lplc_skips_pre_augmented_variants():
    from lvt_eg.data.lplc import parse_lp_filename
    # Should skip filenames with _c_<scale> suffix (paper canonical raw-only)
    assert parse_lp_filename("foo.jpg_0_AAA.jpg_c_2.jpg") is None


def test_list_lplc_folds_order_and_cap(tmp_path: Path):
    from lvt_eg.data.lplc import list_lplc_folds
    for k in range(3):
        (tmp_path / f"fold_{k}.json").write_text("{}")
    folds = list_lplc_folds(tmp_path, repeats=2)
    # k-th entry must be fold_k.json so run_lplc_cv aligns with cv.fold=k.
    assert [f.name for f in folds] == ["fold_0.json", "fold_1.json", "fold_2.json"]


def test_list_lplc_folds_nested_scenario(tmp_path: Path):
    """Source data ships folds under folds/<scenario>/fold_<n>_<iter>.json;
    list_lplc_folds must find them when no flat fold_*.json exist (M3)."""
    from lvt_eg.data.lplc import list_lplc_folds
    scen = tmp_path / "scen0"
    scen.mkdir()
    for n in range(2):
        for it in (1, 2):
            (scen / f"fold_{n}_{it}.json").write_text("{}")
    folds = list_lplc_folds(tmp_path, repeats=2, scenario="scen0")
    assert [f.name for f in folds] == [
        "fold_0_1.json", "fold_0_2.json", "fold_1_1.json", "fold_1_2.json"]


def test_parse_lplc_fold_real_format(tmp_path: Path):
    """parse_lplc_fold must mirror the notebook loader against the REAL shipped
    layout: fold JSON groups by a nested legibility key and stores
    LPRD_Dataset/all_lps/... crop paths, but the loader keys annotations by
    '<image_id>.jpg', loads the FULL scene from images/<image_id>.jpg, and takes
    legibility (readable), OCR, corners from the annotation. This exercises every
    filter (raw-only, dedup, readable, valid, occluded, len==7, ocr cross-check).
    """
    import json

    from lvt_eg.data.lplc import parse_lplc_fold

    xy = [0, 0, 100, 0, 100, 40, 0, 40]
    images = tmp_path / "images"
    images.mkdir()
    # Full-scene image files (existence is all the loader checks).
    for iid in ("a1b2-01", "c3d4-02", "e5f6-03", "0a1b-04", "2c3d-05", "4e5f-06"):
        (images / f"{iid}.jpg").write_bytes(b"")

    annotations = {
        "a1b2-01.jpg": {"anns": [{"ocr": "ABC1D23", "readable": "3",
                                  "valid": True, "occluded": False, "xy": xy}]},   # keep (perfect)
        "c3d4-02.jpg": {"anns": [{"ocr": "GHK4L56", "readable": "1",
                                  "valid": True, "occluded": False, "xy": xy}]},   # keep (poor)
        "e5f6-03.jpg": {"anns": [{"ocr": "OCC9L88", "readable": "2",
                                  "valid": True, "occluded": True, "xy": xy}]},    # drop: occluded
        "0a1b-04.jpg": {"anns": [{"ocr": "INV7D77", "readable": "2",
                                  "valid": False, "occluded": False, "xy": xy}]},  # drop: invalid
        "2c3d-05.jpg": {"anns": [{"ocr": "SHORT", "readable": "3",
                                  "valid": True, "occluded": False, "xy": xy}]},   # drop: len != 7
        "4e5f-06.jpg": {"anns": [{"ocr": "REALL77", "readable": "3",
                                  "valid": True, "occluded": False, "xy": xy}]},   # drop: ocr mismatch
    }
    ann_path = tmp_path / "annotations_formatted.json"
    ann_path.write_text(json.dumps(annotations))

    base = "LPRD_Dataset/all_lps"
    fold = {"test": {
        # nested legibility bucket key is IGNORED - readable comes from annotation
        "0": [
            f"{base}/0/a1b2-01.jpg_0_ABC1D23.jpg",           # keep
            f"{base}/0/a1b2-01.jpg_0_ABC1D23.jpg_c_2.jpg",   # drop: _c_ augmented variant
            f"{base}/0/c3d4-02.jpg_0_GHK4L56.jpg",           # keep
            f"{base}/0/e5f6-03.jpg_0_OCC9L88.jpg",           # drop: occluded
            f"{base}/0/0a1b-04.jpg_0_INV7D77.jpg",           # drop: invalid
            f"{base}/0/2c3d-05.jpg_0_SHORT.jpg",             # drop: len != 7
            f"{base}/0/4e5f-06.jpg_0_ZZZ9Z99.jpg",           # drop: filename ocr != annotation
        ],
    }}
    fold_path = tmp_path / "fold_0_1.json"
    fold_path.write_text(json.dumps(fold))

    samples = parse_lplc_fold(fold_path, ann_path, images, "test", min_legibility=1)

    plates = sorted(s["plate"] for s in samples)
    assert plates == ["ABC1D23", "GHK4L56"]                  # only the two valid readable plates
    keep = next(s for s in samples if s["plate"] == "ABC1D23")
    assert keep["image_path"].replace("\\", "/").endswith("images/a1b2-01.jpg")  # full scene, keyed by id
    assert keep["legibility"] == 3                           # from annotation 'readable', not bucket key
    assert keep["corners"] == [(0, 0), (100, 0), (100, 40), (0, 40)]

    # Passing a preloaded annotations dict must give the identical result.
    assert parse_lplc_fold(fold_path, annotations, images, "test") == samples


def test_ccpd_decode_corners_reorders_to_tl_first():
    """CCPD filenames store corners as [BR, BL, TL, TR]; _decode_corners must
    reorder to crop_plate's expected [TL, TR, BR, BL] so plates crop upright
    (not rotated 180 degrees). Coords from a real CCPD sample."""
    from lvt_eg.data.ccpd import _decode_corners
    # file order: BR(363,554) BL(189,540) TL(190,484) TR(364,498)
    corners = _decode_corners("363&554_189&540_190&484_364&498")
    assert corners == [(190, 484), (364, 498), (363, 554), (189, 540)]  # TL,TR,BR,BL
    assert _decode_corners("1&2_3&4") is None  # wrong count


def test_ccpd_per_subset_grouping():
    from lvt_eg.scripts.evaluate import _ccpd_per_subset
    samples = [
        {"image_path": "/d/ccpd_blur/a.jpg"},
        {"image_path": "/d/ccpd_blur/b.jpg"},
        {"image_path": "/d/ccpd_weather/c.jpg"},
    ]
    preds = ["AB", "XX", "CD"]
    targets = ["AB", "ZZ", "CD"]
    out = _ccpd_per_subset(samples, preds, targets)
    assert out["ccpd_blur"]["n"] == 2
    assert out["ccpd_blur"]["plate_rr"] == 0.5   # 1 of 2 exact
    assert out["ccpd_weather"]["plate_rr"] == 1.0
    assert "ccpd_base" not in out                # empty subset skipped


def test_multiseed_mean_std():
    """run_multiseed aggregation = mean + sample (n-1) std; single value -> std 0."""
    from lvt_eg.scripts.run_multiseed import _mean_std
    m, s = _mean_std([0.90, 0.92, 0.94])
    assert abs(m - 0.92) < 1e-9
    assert abs(s - 0.02) < 1e-9          # sample std of [.90,.92,.94]
    assert _mean_std([0.90]) == (0.90, 0.0)


def test_resolve_lr_aug_policies_build_and_run():
    """Per-dataset LR aug policies resolve; geometric/photometric build and run
    on a dummy image (guards albumentations API drift for the geometric ops);
    'none' -> None; unknown -> ValueError."""
    import numpy as np
    import pytest

    from lvt_eg.data.lp_crop_dataset import resolve_lr_aug
    assert resolve_lr_aug("none") is None
    img = np.zeros((48, 192, 3), dtype=np.uint8)
    for name in ("photometric", "geometric"):
        out = resolve_lr_aug(name)(image=img)["image"]
        assert out.shape == img.shape
    with pytest.raises(ValueError):
        resolve_lr_aug("bogus")


def test_synthetic_dataset_synth_aug_policy():
    """synth_aug='none' disables the extra LR aug on synthetic views (Ukrainian);
    'geometric' keeps a Compose (CCPD)."""
    from lvt_eg.data.synthetic_degradation import MultiLevelSyntheticDataset

    class _Base:
        augment = True
        def __len__(self):
            return 3

    import pytest

    assert MultiLevelSyntheticDataset(_Base(), policy="v2", synth_aug="none").synth_aug is None
    assert MultiLevelSyntheticDataset(_Base(), policy="v2", synth_aug="geometric").synth_aug is not None
    # synth_degrade_at: hr (Ukrainian) vs lr (default); invalid rejected.
    assert MultiLevelSyntheticDataset(_Base(), policy="v2", synth_degrade_at="hr").synth_degrade_at == "hr"
    assert MultiLevelSyntheticDataset(_Base(), policy="v2").synth_degrade_at == "lr"
    with pytest.raises(ValueError):
        MultiLevelSyntheticDataset(_Base(), policy="v2", synth_degrade_at="bogus")


def test_lplc_per_legibility_grouping():
    """LPLC strata breakdown groups by the per-sample 'legibility' level (M1):
    3=perfect, 2=good, 1=poor."""
    from lvt_eg.scripts.evaluate import _lplc_per_legibility
    samples = [{"legibility": 3}, {"legibility": 3}, {"legibility": 2}, {"legibility": 1}]
    preds = ["ABC", "XYZ", "GOOD", "PO"]
    targets = ["ABC", "WWW", "GOOD", "OK"]
    out = _lplc_per_legibility(samples, preds, targets)
    assert out["perfect"]["n"] == 2 and out["perfect"]["plate_rr"] == 0.5
    assert out["good"]["plate_rr"] == 1.0
    assert out["poor"]["plate_rr"] == 0.0


def test_ccpd_decode_plate_positional():
    """_decode_plate maps parts[0]->province, [1]->alphabet, [2:7]->ADS.
    Guards against an off-by-one in the arrays/slice that would silently
    mislabel every CCPD sample."""
    from lvt_eg.data.ccpd import CCPD_PROVINCES, _decode_plate
    plate = _decode_plate("0_0_22_27_27_33_16")
    assert plate is not None and len(plate) == 7
    assert plate[0] == CCPD_PROVINCES[0]   # province index 0
    assert plate[1] == "A"                 # alphabet index 0
    assert plate[2:] == "Y339S"            # ADS indices 22,27,27,33,16
    assert _decode_plate("0_0_1") is None  # fewer than 7 fields


def test_shipped_config_aug_policies():
    """The per-dataset aug fix must survive in the shipped configs: CCPD and
    Ukrainian geometric, Ukrainian synth normalize-only, the rest photometric."""
    from pathlib import Path

    import yaml

    import lvt_eg
    cfgdir = Path(lvt_eg.__file__).resolve().parents[2] / "configs"

    def ds(name):
        return yaml.safe_load((cfgdir / name).read_text())["dataset"]

    ccpd = ds("2-lvt_eg_ccpd.yaml")
    assert ccpd.get("train_aug") == "geometric" and ccpd.get("synth_aug") == "geometric"
    ukr = ds("3-lvt_eg_ukrainian.yaml")
    assert ukr.get("train_aug") == "geometric" and ukr.get("synth_aug") == "none"
    assert ukr.get("synth_degrade_at") == "hr"
    for name in ("1-lvt_eg_rodosol.yaml", "4-lvt_eg_lplc.yaml", "5-lvt_eg_lp2025.yaml"):
        d = ds(name)
        assert d.get("train_aug", "photometric") == "photometric"
        assert d.get("synth_aug", "photometric") == "photometric"


def test_lpcrop_resolves_aug_policy():
    """LPCropDataset stores the resolved LR-aug Compose for its train_aug policy."""
    from lvt_eg.data.lp_crop_dataset import LR_AUG, LR_AUG_GEOMETRIC, LPCropDataset
    assert LPCropDataset([], {}, 64, 256, 48, 192, train_aug="geometric").lr_aug is LR_AUG_GEOMETRIC
    assert LPCropDataset([], {}, 64, 256, 48, 192, train_aug="photometric").lr_aug is LR_AUG
    assert LPCropDataset([], {}, 64, 256, 48, 192).lr_aug is LR_AUG  # default photometric


def test_rodosol_parse_real_fixture(tmp_path: Path):
    """RodoSol parser on a real-format split.txt + sidecar (not just empty-dir)."""
    from lvt_eg.data.rodosol import parse_rodosol_split
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "x.jpg").write_bytes(b"")
    (tmp_path / "images" / "x.txt").write_text("plate: abc1234\ncorners: 1,2 3,4 5,6 7,8\n")
    (tmp_path / "split.txt").write_text("./images/x.jpg;training\n./images/y.jpg;testing\n")
    s = parse_rodosol_split(tmp_path, "training")
    assert len(s) == 1
    assert s[0]["plate"] == "ABC1234"                       # uppercased
    assert s[0]["corners"] == [(1, 2), (3, 4), (5, 6), (7, 8)]
    assert parse_rodosol_split(tmp_path, "testing") == []   # y.jpg absent -> dropped


def test_lp2025_parse_real_fixture(tmp_path: Path):
    """LP-2025 parser: multi-plate line, '_' unreadable skip, coord order."""
    from lvt_eg.data.lp2025 import parse_lp2025_split
    img = tmp_path / "train" / "images"
    lbl = tmp_path / "train" / "labels_gd"
    img.mkdir(parents=True)
    lbl.mkdir(parents=True)
    (img / "1.jpg").write_bytes(b"")
    (lbl / "1.txt").write_text(
        "ABC1234 0 0 10 0 10 4 0 4\n_ 1 1 2 2 3 3 4 4\nAB 0 0 1 0 1 1 0 1\n")
    s = parse_lp2025_split(tmp_path, "train")
    plates = [x["plate"] for x in s]
    assert plates == ["ABC1234", "AB"]                      # readable kept, '_' skipped
    assert s[0]["corners"] == [(0, 0), (10, 0), (10, 4), (0, 4)]


def test_ukrainian_flat_split_deterministic(tmp_path: Path):
    """Flat 80/10/10 auto-split is seed-deterministic (reproduction guarantee)."""
    from lvt_eg.data.ukrainian import parse_ukrainian_dataset
    chars = "0123456789ABCEHIKMOPTXYZ"
    imgs = tmp_path / "images"
    imgs.mkdir()
    for i in range(20):
        (imgs / f"AA{i:04d}AA.png").write_bytes(b"")   # all chars within charset

    def plates(lists):
        return [[s["plate"] for s in part] for part in lists]

    r1 = parse_ukrainian_dataset(tmp_path, chars, seed=42)
    r2 = parse_ukrainian_dataset(tmp_path, chars, seed=42)
    assert plates(r1) == plates(r2)                        # same seed -> same partition
    assert len(r1[0]) == 16 and len(r1[1]) == 2            # 80/10/10 of 20
    r3 = parse_ukrainian_dataset(tmp_path, chars, seed=7)
    assert plates(r3) != plates(r1)                        # different seed -> different partition

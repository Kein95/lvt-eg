"""Smoke test: package and submodules import cleanly."""


def test_package_imports():
    import lvt_eg
    assert lvt_eg.__version__


def test_common_imports():
    from lvt_eg.common import ctc_decode, ema, losses, metrics, seeding, sobel  # noqa: F401


def test_models_imports():
    from lvt_eg.models import (  # noqa: F401
        lvt_eg,
        rctc_decoder,
        stn,
        svtrv2_encoder,
        visual_tactile_branch,
    )


def test_data_imports():
    from lvt_eg.data import (  # noqa: F401
        ccpd,
        lp2025,
        lp_crop_dataset,
        lplc,
        rodosol,
        synthetic_degradation,
        ukrainian,
    )


def test_training_imports():
    from lvt_eg.training import checkpoint, scheduler, trainer  # noqa: F401


def test_scripts_imports():
    from lvt_eg.scripts import (  # noqa: F401
        download_data,
        evaluate,
        run_lplc_cv,
        run_multiseed,
        run_zeroshot,
        train,
    )


def test_dataset_keys():
    from lvt_eg.config import DATASET_SUBDIRS
    assert set(DATASET_SUBDIRS) == {"rodosol", "ccpd2019", "lplc", "ukrainian", "lp2025"}


def test_datahub_files_exist():
    from pathlib import Path

    import lvt_eg
    repo_root = Path(lvt_eg.__file__).resolve().parents[2]
    datahub = repo_root / "data" / "datahub"
    expected = {"1-rodosol-alpr.json", "2-ccpd-2019.json", "3-ukrainian-lp.json",
                "4-lplc.json", "5-lp-2025.json"}
    assert expected.issubset({p.name for p in datahub.glob("*.json")})

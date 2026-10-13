from scripts.build_manifest import build_manifest
from scripts.build_manifest import build_manifest_from_dirs


def test_build_manifest_pairs_images_and_labels(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "case001.nii.gz").write_bytes(b"image")
    (labels / "case001_label.nii.gz").write_bytes(b"label")

    records = build_manifest(
        root=tmp_path,
        dataset="demo",
        relative_to=tmp_path,
    )

    assert len(records) == 1
    assert records[0].case_id == "case001"
    assert records[0].image == "images/case001.nii.gz"
    assert records[0].label == "labels/case001_label.nii.gz"


def test_build_manifest_from_explicit_topcow_dirs(tmp_path):
    images = tmp_path / "image"
    labels = tmp_path / "binary_label"
    multiclass = tmp_path / "multiclass_mask"
    images.mkdir()
    labels.mkdir()
    multiclass.mkdir()
    (images / "example_case_001.npy").write_bytes(b"image")
    (labels / "example_case_001.npy").write_bytes(b"label")
    (multiclass / "example_case_001.npy").write_bytes(b"multiclass")

    records = build_manifest_from_dirs(
        root=tmp_path,
        dataset="topcow",
        relative_to=tmp_path,
        image_dir="image",
        label_dir="binary_label",
        multiclass_label_dir="multiclass_mask",
    )

    assert len(records) == 1
    assert records[0].case_id == "example_case_001"
    assert records[0].image == "image/example_case_001.npy"
    assert records[0].label == "binary_label/example_case_001.npy"
    assert records[0].multiclass_label == "multiclass_mask/example_case_001.npy"

from tcig.interaction import InteractionUnit


def test_click_round_trip():
    unit = InteractionUnit.click((3, 4, 5), polarity=1, round_index=2, metadata={"case": "demo"})
    restored = InteractionUnit.from_dict(unit.to_dict())
    assert restored == unit
    assert restored.center == (3.0, 4.0, 5.0)
    assert restored.is_foreground

import torch

from tcig.clicks import LargestErrorClickSimulator


def test_largest_error_click_simulator_emits_positive_click_for_missing_region():
    target = torch.zeros(8, 8, dtype=torch.long)
    target[2:5, 2:5] = 1
    prediction = torch.zeros_like(target)

    simulator = LargestErrorClickSimulator()
    click = simulator.next_click(prediction, target, round_index=0)

    assert click is not None
    assert click.polarity == 1
    assert click.points[0] == (3, 3)


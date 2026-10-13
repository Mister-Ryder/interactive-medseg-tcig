import json

import numpy as np
import torch

from tcig.data import TopCoWVolumeDataset, collate_topcow_batch, split_manifest_records
from tcig.clicks import LargestErrorClickSimulator
from tcig.model import TCIGSegmenter, TCIGSegmenterConfig
from tcig.train_loop import run_multiround_training_step


def test_split_manifest_records_is_deterministic():
    records = [{"case_id": f"case_{idx}"} for idx in range(10)]
    first = split_manifest_records(records, split="val", val_fraction=0.2, seed=3)
    second = split_manifest_records(records, split="val", val_fraction=0.2, seed=3)
    assert first == second
    assert len(first) == 2


def test_topcow_model_forward_with_empty_and_click_history():
    model = TCIGSegmenter(TCIGSegmenterConfig(embed_dim=16, base_channels=4, scatter_work_resolution_factor=1))
    image = torch.zeros(1, 1, 8, 8, 8)
    empty = model(image, [])
    assert empty["logits"].shape == (1, 1, 8, 8, 8)
    assert empty["trust"].shape == (1, 0)
    click = LargestErrorClickSimulator().next_click(torch.zeros(8, 8, 8), torch.ones(8, 8, 8), round_index=0)
    full = model(image, [click])
    prompt_only = model(image, [click], interaction_mode="prompt_only")
    no_interaction = model(image, [click], interaction_mode="no_interaction")
    assert full["trust"].shape == (1, 1)
    assert prompt_only["trust"].shape == (1, 1)
    assert prompt_only["spatial_context"].abs().sum().item() == 0.0
    assert no_interaction["trust"].shape == (1, 0)


def test_multiround_training_step_runs_on_synthetic_case(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "image").mkdir(parents=True)
    (data_root / "binary_label").mkdir()
    image = np.zeros((8, 8, 8), dtype=np.float32)
    label = np.zeros((8, 8, 8), dtype=np.float32)
    label[2:6, 2:6, 2:6] = 1.0
    np.save(data_root / "image" / "case.npy", image)
    np.save(data_root / "binary_label" / "case.npy", label)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"case_id": "case", "image": "image/case.npy", "label": "binary_label/case.npy"}) + "\n",
        encoding="utf-8",
    )
    dataset = TopCoWVolumeDataset(manifest, data_root, patch_size=(8, 8, 8), split="all")
    batch = collate_topcow_batch([dataset[0]])
    model = TCIGSegmenter(TCIGSegmenterConfig(embed_dim=16, base_channels=4, scatter_work_resolution_factor=1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    row = run_multiround_training_step(
        model=model,
        batch=batch,
        simulator=LargestErrorClickSimulator(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_clicks=2,
    )
    assert row["num_clicks"] >= 1
    assert row["rounds"] >= 2
    assert 0.0 <= row["dice_final"] <= 1.0

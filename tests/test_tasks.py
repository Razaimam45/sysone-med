from med_jevify.tasks.vqarad import image_group_split


def test_vqarad_split_is_image_disjoint():
    rows = [{"image_id": "a", "id": 1}, {"image_id": "a", "id": 2}, {"image_id": "b", "id": 3}, {"image_id": "c", "id": 4}]
    calibration, test = image_group_split(rows, calibration_fraction=0.5, seed=42)
    assert {row["image_id"] for row in calibration}.isdisjoint({row["image_id"] for row in test})
    assert len(calibration) + len(test) == len(rows)

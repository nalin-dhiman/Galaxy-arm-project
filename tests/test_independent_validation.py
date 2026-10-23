"""Protocol checks that prevent held-out labels or WCS flips changing choices."""
import numpy as np
from astropy.wcs import WCS
from experiments.independent_filter_experiment import (label_coordinates,
    map_labels, select_development, paired_interval)


def test_development_choice_ignores_test_and_unknown_controls():
    rows = []
    for identifier in ("a", "b"):
        for pipeline, ap in (("raw", .2), ("filtered", .7)):
            rows.append(dict(id=identifier, split="development", sample_group="strong_spiral",
                             label_votes=3, pipeline=pipeline, average_precision=ap))
    # Deliberately overwhelming held-out/control values cannot choose a filter.
    for split, group in (("test", "strong_spiral"), ("development", "catalog_control")):
        for pipeline, ap in (("raw", 1.), ("filtered", 0.)):
            rows.append(dict(id=split+group, split=split, sample_group=group,
                             label_votes=3, pipeline=pipeline, average_precision=ap))
    assert select_development(rows)[0] == "filtered"


def test_opposite_vertical_wcs_maps_label_to_correct_native_pixel():
    source, target = WCS(naxis=2), WCS(naxis=2)
    for w in (source, target):
        w.wcs.crval = [150., 20.]
        w.wcs.crpix = [6., 6.]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    source.wcs.cdelt = [-.0001, -.0001]
    target.wcs.cdelt = [-.0001, .0001]
    labels = np.zeros((11, 11)); labels[2, 3] = 7
    xs, ys = label_coordinates(source, target, (11, 11))
    mapped, valid = map_labels(labels, xs, ys)
    assert mapped[8, 3] == 7
    assert mapped[2, 3] == 0
    assert valid[8, 3]


def test_bootstrap_uses_galaxies_and_has_no_interval_with_one():
    assert paired_interval([.2])["bootstrap_95_interval"] is None
    result = paired_interval([.1, -.1, .4, -.2])
    assert result["galaxies"] == 4
    assert result["bootstrap_95_interval"][0] < 0 < result["bootstrap_95_interval"][1]

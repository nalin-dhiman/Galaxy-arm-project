import numpy as np
import pytest
from galaxy_lab.kinematic_distance import baryonic_flux_proxy, BaryonicTullyFisher


def test_arbitrary_reference_distance_cancels_from_flux_proxy_and_prediction():
    coefficient = baryonic_flux_proxy(2e10, 3e9, 15.)
    scaled = baryonic_flux_proxy(2e10 * 7**2, 3e9 * 7**2, 15. * 7)
    assert scaled == pytest.approx(coefficient)
    model = BaryonicTullyFisher(10., 4., .1, 30)
    assert model.distance(180, coefficient) == pytest.approx(model.distance(180, scaled))


def test_known_relation_recovers_distances_without_using_target_labels():
    velocities = np.geomspace(35, 300, 30)
    masses = 4e9 * (velocities / 100.) ** 3.7
    model = BaryonicTullyFisher.fit(velocities, masses)
    assert model.slope == pytest.approx(3.7)
    new_v, unknown_distance = 160., 24.
    observed_coefficient = 4e9 * (new_v / 100.) ** 3.7 / unknown_distance**2
    assert model.distance(new_v, observed_coefficient) == pytest.approx(unknown_distance)


@pytest.mark.parametrize("velocity,coefficient", [(0, 1), (np.nan, 1), (100, 0), (100, -1)])
def test_invalid_observables_rejected(velocity, coefficient):
    with pytest.raises(ValueError):
        BaryonicTullyFisher(10, 4, .1, 30).distance(velocity, coefficient)

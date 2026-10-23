"""Empirically calibrated baryonic Tully--Fisher distance, not image-only.

Independent measured rotation speed and externally calibrated baryonic mass
scale are essential. This relation does not identify fluid microphysics.
Mass-to-light, gas calibration, inclination and selection assumptions remain.
Primary example: https://arxiv.org/abs/2006.08615
"""
from dataclasses import dataclass
import numpy as np


def positive(values, name):
    out = np.asarray(values, float)
    if not np.isfinite(out).all() or np.any(out <= 0):
        raise ValueError(name + " must be finite and positive")
    return out


def baryonic_flux_proxy(stellar_luminosity_at_reference, hi_mass_at_reference,
                        reference_distance_mpc, mass_to_light=.5, helium_factor=1.33):
    """Return C=M_b/D_ref² in Msun/Mpc², an observed-flux mass proxy.

    Luminosity is in solar luminosities and HI mass in solar masses. Both
    supplied catalog quantities must have used this *same* reference distance
    quadratically. Dividing an arbitrary dynamical mass by D² is not valid.
    Total stellar luminosity uses one fixed M/L, not a disk/bulge decomposition.
    """
    lum = positive(stellar_luminosity_at_reference, "stellar luminosity")
    gas = np.asarray(hi_mass_at_reference, float)
    if not np.isfinite(gas).all() or np.any(gas < 0):
        raise ValueError("HI mass must be finite and nonnegative")
    distance = positive(reference_distance_mpc, "reference distance")
    upsilon = positive(mass_to_light, "mass to light")
    helium = positive(helium_factor, "helium factor")
    return (upsilon * lum + helium * gas) / distance ** 2


@dataclass(frozen=True)
class BaryonicTullyFisher:
    intercept_log_msun_at_100kms: float
    slope: float
    residual_scatter_dex: float
    calibration_galaxies: int

    @classmethod
    def fit(cls, velocity_kms, baryon_mass_msun, fixed_slope=None):
        velocity = positive(velocity_kms, "velocity")
        mass = positive(baryon_mass_msun, "baryon mass")
        if velocity.ndim != 1 or mass.shape != velocity.shape or len(mass) < 5:
            raise ValueError("At least five matched one-dimensional calibrators required")
        x, y = np.log10(velocity / 100.), np.log10(mass)
        if np.ptp(x) < .05:
            raise ValueError("Calibrators have insufficient velocity leverage")
        if fixed_slope is None:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope = float(positive(fixed_slope, "slope"))
            intercept = float(np.mean(y - slope * x))
        residual = y - (intercept + slope * x)
        scatter = float(np.sqrt(np.sum(residual ** 2) / (len(y) - (2 if fixed_slope is None else 1))))
        return cls(float(intercept), float(slope), scatter, len(mass))

    def mass(self, velocity_kms):
        velocity = positive(velocity_kms, "velocity")
        return 10. ** (self.intercept_log_msun_at_100kms + self.slope * np.log10(velocity / 100.))

    def distance(self, velocity_kms, baryon_flux_msun_per_mpc2):
        """Point estimate in Mpc; no automatically calibrated uncertainty band."""
        return np.sqrt(self.mass(velocity_kms) / positive(baryon_flux_msun_per_mpc2, "flux proxy"))

Derived plotting data: units, provenance and interpretation

publication_plot_data.npz contains55 numeric/boolean arrays and no object arrays.
publication_plot_data.json supplies matching identities, frozen summary numbers
and short schema metadata. provenance.json pins both files by SHA256. Figure
captions, source URLs and visible-credit requirements are in figures.json.

Example arrays

example0 through example3 correspond to MANGAID1-120750,1-460912,1-31587 and
1-156154: the first two strong, first weak/bar-proxy and first unasked entries
in the fixed pre-scoring gallery. Suffixes are:

  _rgb: derived uint8 asinh RGB display from native DR9 z/r/g science planes,
    respectively assigned red/green/blue. This is a display, not new photometry.
  _raw_score: the frozen TRAIN-only classifier's uncalibrated image score,
    dimensionless0to1, conditional on supplied human center/foreground marks.
    It is not a physical-arm posterior probability.
  _valid: boolean common analysis/reference footprint; excluded foreground
    regions and angular analysis limits remain excluded.
  _votes: independent Galaxy Zoo3D polygon vote count, mapped through celestial
    WCS to the native grid. Counts are absolute votes, not a universal fraction.
    Pixels with at least3votes form the displayed positive reference region.
  _extent: image-frame offsets in arcsec. They are not physical lengths.

The first example has no two-class consensus; it was not silently removed.
The final example has no spatial arm task: truth is unknown, never negative.
White valid unmarked reference pixels can still contain incompletely traced
arms. Native color data, PSFs and masks originate from the cited science
products; no missing band was fabricated or inferred from a JPEG display.

Legacy Surveys data and required visible credit:
  Legacy Surveys / D. Lang (Perimeter Institute)
  https://www.legacysurvey.org/acknowledgment/
Galaxy Zoo3D independent human regions/marks:
  Masters et al.2021, Galaxy Zoo:3D, https://arxiv.org/abs/2108.02065
  https://www.sdss4.org/dr17/manga/manga-data/manga-gz3d/
  https://www.sdss4.org/collaboration/citing-sdss/

Redshift arrays and summary data

redshift_z, redshift_PGI and redshift_PGIA are the inherited catalogue redshift
and fixed estimates on all59 additional strong-spiral identities. The integer
redshift_missing flag preserves the imputed missing-image row. The JSON stores
original29 and fresh59 paired effects separately, with explicitly equal-sky-
group weighting and100000-draw grouped uncertainty. AP summaries use their own
stated estimands: original paired gains average sky groups equally; fresh AP
is galaxy-macro with whole-group resampling. The controls L5/L3 use only the
original91 training galaxies and are retrospective feature comparisons. The
source literature motivation is Soo et al.2018, Morpho-z, MNRAS475,3613:
  https://doi.org/10.1093/mnras/stx3201
These numbers do not reproduce that paper's ANN pipeline, CS82 data or selection.
Catalogue-source description:
  https://www.sdss4.org/dr17/manga/manga-tutorials/drpall/

Independent centerline arrays

geometry0 and geometry1 correspond to the first two fixed held-out Galaxy
Builder subjects21096863 and21096986. _data is source-normalized linear SDSSr
intensity, _valid is the source mask complement, and _extent gives angular
image-frame coordinates at0.396arcsec per native pixel. _ref0,etc. are derived
author-cleaned/clustered logarithmic human-fit centerline coordinates in arcsec;
_pred0,etc. are the fixed template predictions in the same frame. These are
plot coordinates, not raw classification records or executable author objects.
The renderer does not refit them. No celestial WCS or absolute photometric
calibration is supplied for this task. Human disk geometry is external input.
Credit SDSS, Galaxy Builder volunteers, and Lingard et al.2020/2021:
  https://arxiv.org/abs/2006.10450
  https://arxiv.org/abs/2105.04500
  https://github.com/tingard/hierarchical-modelling-of-spiral-pitch-angle
  https://data.galaxyzoo.org/

Finite-equilibrium arrays

halo_radius is equatorial kpc. halo_density is solar masses/kpc^3, halo_pressure
is density*(km/s)^2, halo_sigma2 is local one-dimensional velocity variance in
(km/s)^2, and halo_velocity is total circular speed in km/s. These are sampled
forward-model curves at stated parameters, not a new fit to a galaxy. Parameters
and resolution are recorded in the JSON. Density and pressure vanish at the
physical free surface; vacuum has no halo velocity dispersion and its cached
zero placeholder is not plotted as a zero-temperature halo. Static equilibrium
is compatible with an established collisionless distribution and does not
identify viscosity or scattering. The cached curve does not break the absolute
distance degeneracy.

Baryonic Tully-Fisher arrays

btfr_reference, btfr_reference_error, btfr_distance, btfr_lower90 and btfr_upper90
are Mpc values on the same30 local calibrators. Each distance prediction is
leave-one-galaxy-out; intervals are conditional90% model intervals, not an
independent calibration guarantee. btfr_extrapolation flags two speed-range
extrapolations, retained in the display. Matching galaxy names are in JSON.
The median interval width divided by PREDICTED distance is0.8220939773; divided
by REFERENCE distance it is0.8414188789. No normalization ambiguity is hidden.
Published source data and source acknowledgments:
  https://astroweb.case.edu/SPARC/
  https://astroweb.case.edu/SPARC/BTFR_Lelli2019.mrt
The renderer/cache does not contain raw IR/HI images or geometric distances
inferred from arms. Shared anchors and population-selection limitations remain.

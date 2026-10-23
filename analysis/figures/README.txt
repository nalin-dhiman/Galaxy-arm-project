Portable reconstruction of the nine paper figures

This folder is self-contained. It needs Python3, NumPy and Matplotlib, including
Matplotlib's usual dependencies. It does not import the research workspace,
Astropy, pandas, scikit-learn, serialized models, or downloaded author code.
It does not make network requests. The compressed plot-array archive is under
1MB. The files contain derived plotting material, not raw FITS archives or
Python object pickles.

Install and run:

  python -m pip install -r requirements.txt
  python render_figures.py

Or choose an output directory:

  python -I render_figures.py --output rendered

Nine standalone PDFs and PNGs, figure_qa.json and an output figures.json appear
in the output directory. The renderer checks the data hashes in provenance.json
and uses np.load(..., allow_pickle=False); object arrays are prohibited. PDF
timestamps and font/rendering differences across library versions can change
file bytes. No cross-platform byte-identity claim is made.

The exact tested versions are recorded in tested_versions.json. The isolated replay is recorded in portable_verification.json. That run used
Python's isolated mode, outside the research workspace. All nine rendered PNG
pixel arrays matched the final publication figures in the tested environment;
all nine figures had zero detected rendered-text bounding-box collisions.
The test report distinguishes this display reproduction from scientific
reproduction of training, raw-data reduction or statistical inference.

Important scope

The cache contains already derived RGB composites, raw classifier scores,
evaluation footprints, WCS-mapped vote masks, explicit angular centerlines,
sampled equilibrium profiles and frozen statistical summaries/predictions.
The renderer faithfully draws these values; it neither refits a model nor
recreates survey measurements. Repeating this renderer is not a fresh model
validation or new scientific evidence. Full scientific reproduction additionally
needs the cited source data, selection rules, processing and analysis code.

The image/reference identities follow the fixed pre-scoring gallery. No figure
selects a newly successful test object. The old and fresh redshift contrasts are
separate already observed cohorts. The practical-effect margin is retrospective.
The closest-feature morphology controls are exploratory, not an exact ANN/CS82
reproduction. No absolute distance ruler or viscosity measurement is inferred.

Presentation corrections

The gallery now explicitly distinguishes gray excluded pixels, white valid
unmarked reference pixels and green regions with at least three human votes.
Unmarked reference pixels are not established physical negatives. The BTFR
caption's median interval-width denominator is predicted distance (82.21%),
not reference distance (84.14%). These are presentation clarifications; the
scientific observations, predictions, intervals and previous frozen results
were not changed. See presentation_corrections.json.

Attribution and data rights

The renderer and its original plotting code are separate from the underlying
survey material. This folder does not relicense survey images or volunteer
annotations. Keep all visible image credits and the source citations in any
scientific reuse. Legacy Surveys images retain their CC BY4.0 attribution
requirements; SDSS and Galaxy Zoo/Builder material retains its source policies.
The NPZ cache is for the attributed scientific plots, not a newly licensed bulk
training dataset. DATA_README.txt identifies the arrays and original sources.

Do not add this image cache to an Overleaf-only source package unnecessarily;
the compiled figure PDFs already suffice to build the paper. A repository can
publish this small renderer/cache separately for transparent display replay.

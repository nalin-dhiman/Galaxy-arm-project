Galaxy Arm Project
==================

Tools for measuring spiral structure and testing the information it contributes
to galaxy redshift estimates. The project combines calibrated multiband image
analysis, saved arm-region models, grouped statistical comparisons, and finite
halo-equilibrium calculations.

Use the application
-------------------

The browser workbench is published at
https://nalin-dhiman.github.io/Galaxy-arm-project/ . It provides:

* Local image upload, grayscale channels and spatial filters.
* Two-point pixel/angular separation and physical separation when an external
  angular-diameter distance is supplied.
* Manual arm tracing, logarithmic pitch fits and same-ray radial gaps, with
  supplied disk-center, inclination and image-angle geometry.
* A numerical spherical lowered-isothermal halo explorer.
* An interactive view of the frozen observational results.

Images are processed in the browser. The Python library supplies the calibrated
FITS, PSF, learned-model and spectral-analysis workflows. Browser RGB channels
are display channels; calibrated survey passbands require the Python workflow.

Install the analysis software
----------------------------

Use Python 3.12 for the recorded model environment::

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-repro.txt
    pip install -e . --no-deps

For a regular library installation, ``pip install -e '.[test]'`` uses the
supported dependency bounds. The bundled learned models were created with
scikit-learn 1.6.1; ``requirements-repro.txt`` records the working environment.

Run a real calibrated-image example::

    python examples/run_native_example.py --output-dir outputs/native_example

This uses the included Legacy Survey grz cutout, local PSF, observed photometry,
and supplied foreground mask. It writes maps, predictors, a catalog-redshift
estimate and a report with input-scope checks. Model metadata records the
training population and calibration assumptions.

Other measurements::

    galaxy-lab pair --ra1 0 --dec1 0 --ra2 0.0002777777778 --dec2 0
    galaxy-lab fluid --rho0 10000000 --sigma 100 --radii 0 1 5 10 30
    python -m galaxy_lab.finite_fluid --radii 1 5 10 50 150
    galaxy-lab filter --help
    galaxy-lab kinematics --help
    galaxy-lab btfr-distance --help

Reproduce numerical comparisons
------------------------------

The ``results/`` directory supplies the frozen galaxy-level predictions,
grouped effects, observable feature tables, and an explicit data card. Run::

    python analysis/reproduce_tables.py --output-dir outputs/reproduced_tables
    python analysis/reproduce_tables.py --refit-controls --output-dir outputs/refit_controls

The second command refits the four literature-motivated structural controls
using only the original 91 training galaxies and compares their predictions
with the saved 59-galaxy table. Training arm descriptors in that table were
generated out of fold by sky group. These controls are retrospective feature
adaptations, not a reproduction of another paper's survey or neural estimator.

``experiments/`` contains the image acquisition, segmentation, kinematic,
equilibrium and validation drivers used for the larger research workflows.
Some drivers expect their documented survey caches and historical acquisition
directory layouts. The compact ``analysis/`` reproduction works directly with
the tables included here. Raw survey caches can be reacquired from the recorded
public services; they are not required to reproduce the tabulated comparisons.

Scope of the evidence
---------------------

The classifier improves arm-region ranking relative to a matched local
brightness/radius control. Its eight arm summaries have not established an
incremental redshift gain in the two reported test cohorts. The target is
inherited NSA catalog redshift, rather than a geometric distance. The finite
halo implementation is an equilibrium model based on established physics;
viscosity is not identified by those static fits. Model cards and the results
data card record the populations, calibration assumptions and measurement scope.

Check the software
------------------

Run the Python tests and dependency-free browser math tests::

    python -m pytest -q
    node --test web/tests/*.test.js

To run the browser app locally::

    python -m http.server 8000 --directory web/dist

Then open http://localhost:8000 . The static browser application is published through the ``gh-pages`` branch.
Ready-to-use check/deployment workflow templates are supplied in
``automation/github-actions/`` for accounts with workflow-write permission.

Repository contents
-------------------

``galaxy_lab/`` contains the complete maintained Python library; ``models/``
contains its saved arm, redshift and calibrated-kinematic artifacts; ``tests/``
contains the numerical and interface tests; ``examples/`` contains a usable
calibrated example; ``results/`` contains frozen numerical evidence; and
``web/`` contains the browser application.

Repository scope
----------------

This repository distributes the application, analysis library, examples and
reproducible numerical outputs. Generated reports and manuscripts are not part
of the distribution. The browser deployment contains the static application
and its compact numerical data; build logs and test artifacts are excluded.

Credits and reuse
-----------------

See ``THIRD_PARTY_NOTICES.txt`` for survey, citizen-science, algorithm and
dependency credits and source links. Legacy imagery is credited
to Legacy Surveys / D. Lang (Perimeter Institute). Public source availability
does not change the terms of third-party data. The project owner's software
licence selection is recorded separately when supplied.

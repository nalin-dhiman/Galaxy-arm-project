READING AND BUILDING THE PAPER

main.tex is the main article. supplementary.tex is a standalone detailed
supplement. references.bib is shared. paper_setup.tex contains ordinary
portable LaTeX packages; author.tex is the single editable author configuration.
The figures/ directory contains PDF figures for publication and PNG previews.
reference_sources.json records the primary literature links and where each
reference is cited.

Read the main paper first for the question, mathematical measurement model,
positive arm-region results, and the controlled redshift comparison. The
supplement supplies exact features, sampling, PSF/noise assumptions, grouped
inference, reference conventions, and the separate physical calculations.

On Overleaf, upload the source bundle, select main.tex as the main document and
use pdfLaTeX. Select supplementary.tex as the main document to build the
supplement. No local absolute paths, shell escape, custom journal class or
network access is needed for the LaTeX build.

Equivalent local commands, run from this directory:

  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex

Repeat with supplementary in place of main for the supplement.

AUTHOR INFORMATION

The editable author configuration defaults to Nalin Dhiman without an invented
affiliation. Confirm the complete author list, contributions, affiliations and
author approval before submission. A repository account does not establish
coauthor consent. Add any journal-specific disclosure or acknowledgment required
for the final submission. These checks are not claims in the scientific paper.

SOURCE AND SCIENTIFIC SCOPE

Companion repository: https://github.com/nalin-dhiman/Galaxy-arm-project
No journal acceptance or independent external review is implied by this bundle.
The main reported positive result is improved agreement with human arm-region
annotations. The incremental redshift result remains inconclusive; do not
remove it when adapting the article. Centerline validation and kinematic/halo
calculations are separate tasks. The dynamics use independent velocities and
external mass/distance assumptions; no viscosity is measured from the images.

References are genuine literature sources with publisher, arXiv or original
paper links. The 2026 ZooBot:3D entry is identified as a preprint. Each included
reference is cited in the main article or supplement; the bibliography is not
an asserted ranking of alternatives that were not run.

All figures retain source-specific credits. In particular, Legacy-derived
images retain the visible credit 'Legacy Surveys / D. Lang (Perimeter Institute)'.
Archive reuse terms and project source/model terms are distinct. Do not infer
a blanket third-party data license from public availability of this repository.

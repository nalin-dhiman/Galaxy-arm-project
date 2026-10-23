Galaxy Structure Lab — browser workbench

Deploy the contents of dist/ as a static site. There is no build step, package installation, CDN or server-side image upload. Serve through HTTP(S); ES modules and the compact local result table require a normal web origin.

Local preview:
python -m http.server 8765 --bind 127.0.0.1 --directory web/dist
Open http://127.0.0.1:8765/

Checks:
node web/tests/science.test.js
node web/tests/integration.test.js
python web/tests/browser_integration.py
The last check uses an installed Python Playwright/Chromium and a running local preview. It exercises real uploads, filters, keyboard selection, measurements, controls, all four exports, and desktop/mobile layouts. Screenshots and test artifacts go to test_output/; they are not needed for deployment.

Image workbench:
Open a local PNG or JPEG. Images remain in the browser. Display channels, grayscale, Gaussian smooth, unsharp, gradient and Hessian bright-ridge filters operate on encoded display colors, not calibrated astronomical flux or g/r/z science passbands. The initial example is an analytically generated, clearly labeled synthetic spiral with seeded noise. Uploading a real image resets its pixel scale and external distance. Images larger than 1,200 pixels on one side are resampled for processing; point coordinates map back to the original image grid. Two-point separations use the supplied original pixel scale. Physical separation is a small-angle projection L = D_A theta using an independently supplied angular diameter distance. No redshift or distance is inferred from the image. Arrow keys and Enter also select measurement points. PNG export excludes point overlays; JSON exports calibration and original coordinates.

Spherical finite halo:
The browser integrates the known lowered-isothermal (King) spherical equilibrium using RK4 with a regular central start and the first W=0 edge. It uses the same density and pressure moments as the Python research implementation; the positive series avoids edge cancellation. x=r/r0, r0=s/sqrt(4 pi G rho0), dW/dx=-m/x^2, dm/dx=x^2 K_density(W)/K_density(W0), rho=rho0 K_density(W)/K_density(W0), P=rho0 s^2 K_pressure(W)/K_density(W0). Circular speed follows enclosed mass and continues as a Keplerian exterior field. Density and pressure vanish outside the edge. Controls are bounded to W0=1..8. Tests compare edge radius and total mass against independent Python integrations and check pressure derivative, hydrostatic equilibrium and physical scaling. This is a spherical halo without baryons; it is not the full axisymmetric disk-halo solver, a fit to a selected galaxy, or a viscosity measurement. Pressure units are M_sun/kpc^3*(km/s)^2. Velocity scale s is not a spatially constant local velocity dispersion.

Evidence explorer:
results.json contains sanitized frozen per-object arm-region scores from the previously evaluated fresh 99-galaxy, same-survey cohort. Selection changes only the displayed population or independent human consensus threshold (3 or 5 votes), not the trained model or its candidate threshold. The fixed candidate threshold is 0.30. All 99 identities remain in the source research evaluation: 98 images passed input guards, 79 galaxies have two-class consensus at 3 votes, and 16 low-vote objects had no spatial arm task and therefore supply unknown rather than negative truth. Primary strong-spiral AP is 0.689034 across 57 evaluable of 59 selected identities. Undefined metrics export blank. The displayed catalog-redshift comparison has a sky-group improvement interval spanning zero, so it does not establish an incremental distance benefit from arms. These aggregate data do not permit arbitrary model-threshold retuning or centerline/pitch validation.

Public resources:
The page links to the public GitHub repository and the paper, supplement and Overleaf source bundle under downloads/.

Primary scientific references:
King (1966), https://doi.org/10.1086/109857
Gieles & Zocchi (2015), https://arxiv.org/abs/1508.02120
The public project repository contains the full Python solver, reproducible experiments, provenance and limitations.

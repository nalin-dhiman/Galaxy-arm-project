GitHub Actions templates

The YAML files under github-actions/ define scientific checks and optional
Actions-based Pages deployment. To activate them, place them under
.github/workflows/ using an account credential with workflow-write permission.
The currently published app uses the gh-pages branch. The test template checks
the numerical library and browser application; the deployment template publishes
only the static application assets under web/dist/.

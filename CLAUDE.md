# biophys_helpers

## Dependencies

This project's environment is defined in `environment.yaml`.

Whenever you add or reference a new third-party dependency, add it to
`environment.yaml` in the same change. This includes:

- A new `import` of a package not already listed in the env.
- A new engine/backend that an existing library requires — e.g. `openpyxl`
  for pandas `.xlsx` I/O, which pandas treats as an optional extra and does
  not pull in on its own.

Guidelines:

- Add it as a conda dependency by default. Use a `pip:` subsection only if the
  package isn't available on the configured conda channels.
- Don't remove existing dependencies without first confirming nothing in the
  codebase still imports them.

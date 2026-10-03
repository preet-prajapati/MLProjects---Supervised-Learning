# Contributing

Create a branch, install `requirements-dev.txt`, and keep changes focused on a specific problem.
For a new project, add a guide, metadata, data acquisition instructions, loader contract,
model pipeline, notebook, and meaningful tests. Never commit private data, credentials, editor caches,
or serialized models from untrusted sources.

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q
python scripts/check_notebooks.py
python scripts/check_notebooks.py --execute bundled
```

For housing changes, explicitly download the dataset and execute its notebook too. State any
validation that could not run. Update reports only after real execution and explain methodology
changes that make scores incomparable. Keep raw data immutable and document its source and rights.

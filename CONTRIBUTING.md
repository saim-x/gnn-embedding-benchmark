# Contributing

Thanks for your interest in improving this benchmark scaffold.

## Getting started

1. Fork the repository and create a feature branch.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run quality checks before opening a PR:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```

## Contribution standards

- Keep changes scoped and reproducible.
- Preserve citation anchors and paper-reference comments where relevant.
- Document any new assumptions in `REPRODUCTION_NOTES.md`.
- Update `README.md` when behavior or usage changes.

## Pull request checklist

- [ ] Tests pass locally.
- [ ] New behavior is documented.
- [ ] Reproduction assumptions are updated if needed.
- [ ] No large datasets or model artifacts are committed.


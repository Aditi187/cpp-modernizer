# Contributing to C++ Modernizer

First off, thank you for considering contributing to the C++ Modernizer Engine! It's people like you that make open-source such a fantastic community.

## Development Setup

1. Fork the repo and create your branch from `main`.
2. Install Python 3.12+.
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
4. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, if applicable.
3. Your code must pass all formatting and linting checks (`black`, `flake8`).
4. You may merge the Pull Request in once you have the sign-off of at least one other developer.

## Reporting Bugs

Please use the Bug Report issue template to report any bugs. Ensure you include:
- A clear and descriptive title.
- Exact steps to reproduce the problem.
- Expected vs. actual behavior.
- Context (OS, Python version, compiler version).

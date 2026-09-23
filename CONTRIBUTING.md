# Contributing to MarsWalk

Thank you for your interest in contributing to **MarsWalk**! We welcome contributions from planetary scientists, software engineers, roboticists, and space exploration enthusiasts.

---

## Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

## Getting Started

1. **Fork the Repository**: Create a personal fork on GitHub.
2. **Clone your Fork**:
   ```bash
   git clone https://github.com/<your-username>/marswalk-nasa.git
   cd marswalk-nasa
   ```
3. **Set Up Python Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-cool-feature
   ```

---

## Development Guidelines

- **Code Style**: Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines for Python code.
- **Large Data**: Never commit `.tif`, `.npy`, or files larger than 10 MB directly to git. Use `scripts/fetch_big.py` and `scripts/fetch_data.py` to stream datasets on demand.
- **Secrets & Credentials**: Ensure `.env` is used for sensitive variables. Never commit tokens, passwords, or private keys.
- **Tests & Validation**: Before opening a Pull Request, run the validation suite:
   ```bash
   python scripts/run_validation.py
   ```

---

## Submitting Pull Requests

1. Commit your changes with clear, descriptive commit messages following the Conventional Commits specification (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
2. Push your branch to your fork:
   ```bash
   git push origin feature/my-cool-feature
   ```
3. Open a Pull Request against the `main` branch.
4. Fill out the PR template with description, testing steps, and relevant screenshots.

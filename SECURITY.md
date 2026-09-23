# Security Policy

## Supported Versions

MarsWalk follows semantic versioning and actively maintains the latest release branch for security and operational updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

The MarsWalk team takes the security and integrity of our software and mission data seriously. If you identify a security vulnerability or sensitive data leak, please report it responsibly.

### How to Report

1. **Do NOT file a public issue** for security vulnerabilities or credential leaks.
2. Email the maintainer directly at **antrikshyadav97@gmail.com** with the subject line:  
   `[SECURITY VULNERABILITY] MarsWalk - <brief description>`.
3. Provide detailed steps to reproduce the issue:
   - Component or endpoint affected (e.g., FastAPI route, data ingestion, telemetry proxy).
   - Proof of Concept (PoC) or reproduction steps.
   - Potential impact of the issue.

### Response & Disclosure Process

- **Acknowledgment**: You will receive an acknowledgment within 48 hours.
- **Triage & Remediation**: We will investigate, confirm, and prepare a patch within 7 business days.
- **Coordinated Disclosure**: A public advisory and patched release will be issued once the fix is deployed.

---

## Security Practices

### Credentials & Secret Management
- **Zero Secrets in Code**: No API keys, NASA personal access tokens, or credentials may ever be committed to the repository.
- Use local environment variables (`.env` ignored by `.gitignore`) for private tokens.
- Automated secret scanning is enforced on all pull requests via GitHub Actions.

### Data Provenance & Integrity
- All external satellite mosaics and elevation models (DTM / DEM) fetched from USGS Astrogeology are verified against published SHA-256 cryptographic hashes before ingestion.

# Security Policy — Neura-X

**Intelligence Without Limits.** Founded by Edusei Mikel Lisamba.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Yes    |
| < 1.0   | ❌ No     |

## Reporting a Vulnerability

Please report security vulnerabilities **privately** by email:

- **Email:** lisambamikel@gmail.com
- **Subject:** `[Neura-X] Security — <short summary>`

Include: description, affected component (C / C++ / Rust / Python / .nex
format), reproduction steps, and impact assessment.

**Do not** open public issues or discussions for security vulnerabilities.

## Response Commitment

- Acknowledgement within **72 hours**.
- Status update within **7 days**.
- Coordinated disclosure after a fix is released.
- Credit in the release notes (unless you request anonymity).

## Integrity Guarantees

- The **Founder's Lock** (SHA-256 signature in every `.nex` file) is a
  security boundary. Attempts to bypass, forge, or strip it are treated as
  security incidents and license violations.
- Model provenance: always verify `nx._core.get_founders_lock_signature()`
  against published values before trusting third-party `.nex` artifacts.

## Safe Harbor

Good-faith research conducted in accordance with this policy is authorized.

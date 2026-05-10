# Security Policy

## Supported Versions

Only the latest minor release receives security updates. Older versions receive no
backports — please upgrade to the latest release.

| Version | Supported |
|---------|-----------|
| 1.x.x   | ✓         |
| < 1.0   | ✗         |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report vulnerabilities privately via GitHub's [Security Advisories](https://github.com/Fulton-Engineering-Services/bge-m3-embedding-server/security/advisories/new).

You can expect:
- Acknowledgement within 48 hours
- A fix or mitigation timeline within 7 days for critical vulnerabilities
- Credit in the release notes (unless you prefer to remain anonymous)

## Scope

This service processes text inputs to generate embeddings. Key attack surface:
- HTTP endpoint input validation (batch size, string length, content type)
- Docker image base image vulnerabilities
- Rust dependency supply chain (monitored via `cargo deny` and Dependabot)

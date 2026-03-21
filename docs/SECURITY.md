# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in q2netcdf, please report it responsibly.

### How to Report

**Please DO NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security vulnerabilities by email to:

**Email**: pat@mousebrains.com

### What to Include

When reporting a vulnerability, please provide:

1. **Description**: A clear description of the vulnerability
2. **Impact**: What could an attacker accomplish by exploiting this?
3. **Reproduction**: Step-by-step instructions to reproduce the issue
4. **Environment**: Python version, OS, q2netcdf version
5. **Proof of Concept**: If applicable, sample code demonstrating the vulnerability
6. **Suggested Fix**: If you have ideas on how to fix it (optional)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Assessment**: We will assess the vulnerability and determine its severity
- **Timeline**: We will provide an estimated timeline for a fix
- **Updates**: We will keep you informed of our progress
- **Credit**: With your permission, we will credit you in the security advisory

### Security Considerations for Q-file Processing

When working with Q-files from untrusted sources, be aware:

1. **File Size**: Q-files can be very large. The `mergeqfiles` tool has size limits, but be cautious when processing files from unknown sources
2. **Binary Parsing**: This library parses binary data. Malformed files could potentially cause unexpected behavior
3. **Configuration Files**: JSON configuration files for `QReduce` are parsed with standard Python libraries. Use caution with configs from untrusted sources
4. **Command Injection**: When using this library in scripts, be careful not to pass user input directly to shell commands

### Best Practices

- Always validate file sizes before processing
- Use the latest version of q2netcdf
- Keep Python and dependencies up to date
- Run with minimal privileges when processing untrusted files
- Consider sandboxing when processing files from unknown sources

## Disclosure Policy

- We follow responsible disclosure practices
- Security issues will be patched as quickly as possible
- Once a fix is released, we will publish a security advisory
- We will credit researchers who report vulnerabilities (with permission)

## Scope

This security policy applies to:
- The q2netcdf Python package
- Command-line tools (q2netcdf, mergeqfiles, QHeader, etc.)
- Example code in the repository

This policy does not cover:
- Q-file format vulnerabilities (contact Rockland Scientific)
- Issues in dependencies (report to the respective projects)
- Issues in user-written code that uses q2netcdf

## Contact

For security issues: pat@mousebrains.com

For general questions: Use [GitHub Issues](https://github.com/mousebrains/q2netcdf/issues) or [GitHub Discussions](https://github.com/mousebrains/q2netcdf/discussions)

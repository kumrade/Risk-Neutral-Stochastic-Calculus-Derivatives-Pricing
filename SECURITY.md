# Security notes

- Never commit SmartAPI credentials, PINs, TOTP secrets, session tokens or generated log files.
- Use environment variables for the command-line smoke test.
- Credentials entered through the desktop interface remain in process memory for the active session.
- Review screenshots before publication to ensure that account identifiers and credentials are not visible.
- Revoke and rotate any credential that has been accidentally exposed.

# Security

Open Asset Studio is a single-user tool for a trusted machine or network.

- **No authentication.** Anyone who can reach the port can run jobs, read and delete assets, and change settings.
  The default host is `127.0.0.1`. Use `--host 0.0.0.0` only on a network you trust, or put it behind a reverse
  proxy with authentication.
- **Models run arbitrary upstream code.** Several backends load models with `trust_remote_code` or import
  third-party repositories from `vendor/`. Only install backends and weights from sources you trust.
- **Cloud keys** (optional) are stored in `data/secrets.json` with mode 600. `data/` is git-ignored.
- **Uploads** are validated by content (images, audio, GLB, video via magic bytes and a locked-down `ffprobe`),
  and files are served only from inside each asset folder. Reports of bypasses are very welcome.

## Reporting a vulnerability

Please report it privately with a GitHub security advisory ("Report a vulnerability" on the repository's
Security tab), not in a public issue.

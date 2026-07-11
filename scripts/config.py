#!/usr/bin/env python3
"""
Manage the PRODUCTION instance config that lives on the fly.io volume
(/app/state/config.yaml) over fly's authenticated SSH. No HTTP, no public surface.

Workflow:
    python scripts/config.py pull              # download prod config -> ./config.yaml
    # edit ./config.yaml locally ...
    python scripts/config.py push              # validate -> upload -> restart to apply
    python scripts/config.py diff              # show local vs prod differences
    python scripts/config.py edit              # pull -> open $EDITOR -> push
    python scripts/config.py restart           # restart the app (reload config)

Options:
    --app    fly app name      (env FLY_APP,    default "signalautomation")
    --file   local config file (default "config.yaml")
    --remote remote path       (default "/app/state/config.yaml")
    --yes    skip the confirm prompt on push/restart

Requires: flyctl on PATH and `flyctl auth login` already done.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Make `import src.config` work when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP_DEFAULT = os.getenv("FLY_APP", "signalautomation")
REMOTE_DEFAULT = "/app/state/config.yaml"


def fly(*args: str) -> None:
    """Run a flyctl command, streaming output; exit on failure."""
    cmd = ["flyctl", *args]
    print("  $ " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        sys.exit(f"ERROR: command failed: {' '.join(cmd)}")


def validate(path: str) -> None:
    """Parse + build the config exactly like the server would, so we never upload
    something that would crash the app on restart. Uses the local .env for ${VAR}."""
    try:
        from src.config import load_config
        load_config(Path(path))
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"ERROR: {path} is invalid -- not uploading:\n  {exc}")
    print(f"OK: {path} validated.")


def confirm(msg: str, skip: bool) -> None:
    if skip:
        return
    if input(f"{msg} [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("Aborted.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["pull", "push", "diff", "edit", "restart"])
    p.add_argument("--app", default=APP_DEFAULT)
    p.add_argument("--file", default="config.yaml")
    p.add_argument("--remote", default=REMOTE_DEFAULT)
    p.add_argument("--yes", action="store_true", help="skip confirmation prompts")
    args = p.parse_args()

    def pull_to(dest: str) -> None:
        fly("ssh", "sftp", "get", args.remote, dest, "-a", args.app)

    def push_from(src: str) -> None:
        validate(src)
        confirm(f"Upload {src} -> {args.app}:{args.remote} and RESTART?", args.yes)
        # fly sftp won't overwrite existing files, so upload to <remote>.new then atomically move into place
        tmp_remote = args.remote + ".new"
        fly("ssh", "console", "-a", args.app, "-C", f"rm -f {tmp_remote}")
        fly("ssh", "sftp", "put", src, tmp_remote, "-a", args.app)
        fly("ssh", "console", "-a", args.app, "-C", f"mv -f {tmp_remote} {args.remote}")
        fly("apps", "restart", args.app)
        print("OK: uploaded + restart triggered. Watch logs: flyctl logs -a " + args.app)

    if args.command == "pull":
        pull_to(args.file)
        print(f"OK: pulled {args.app}:{args.remote} -> {args.file}")

    elif args.command == "push":
        if not os.path.exists(args.file):
            sys.exit(f"ERROR: {args.file} not found")
        push_from(args.file)

    elif args.command == "diff":
        tmp = Path(tempfile.gettempdir()) / "config_prod.yaml"
        pull_to(str(tmp))
        local = Path(args.file).read_text(encoding="utf-8").splitlines()
        prod = tmp.read_text(encoding="utf-8").splitlines()
        import difflib
        delta = list(difflib.unified_diff(prod, local, fromfile="prod", tofile="local", lineterm=""))
        print("\n".join(delta) if delta else "No differences (local == prod).")

    elif args.command == "edit":
        pull_to(args.file)
        editor = os.getenv("EDITOR") or ("notepad" if os.name == "nt" else "vi")
        subprocess.call([editor, args.file])
        push_from(args.file)

    elif args.command == "restart":
        confirm(f"Restart {args.app}?", args.yes)
        fly("apps", "restart", args.app)


if __name__ == "__main__":
    main()

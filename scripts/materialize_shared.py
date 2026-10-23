"""Materialize the pinned periodic submodule using repository configuration."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def materialize(root):
    config = json.loads(
        (root / "scripts/materialize-config.json").read_text(encoding="utf-8")
    )
    source = root / "_periodic_shared/periodic_shared"
    if not source.is_dir():
        raise FileNotFoundError(
            "Run git submodule update --init before materializing shared modules."
        )
    package = (root / config["package"]).resolve()
    package.relative_to(root.resolve())
    if not package.is_dir():
        raise FileNotFoundError(package)
    for name in config["files"]:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError(f"Invalid shared module filename: {name}")
        (package / name).write_bytes((source / name).read_bytes())
        print(name)


def files_to_copy():
    config = json.loads(
        (ROOT / "scripts/materialize-config.json").read_text(encoding="utf-8")
    )
    return list(config["files"])


def main():
    materialize(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

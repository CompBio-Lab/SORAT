"""Command-line entry point: ``python -m hcmv <command> [--config ...] [--set key=value ...]``.

Commands are registered in ``COMMANDS`` as tickets land; each receives the loaded
config and the parsed arguments.
"""

import argparse
import json
import sys

from . import __version__
from .config import config_hash, load_config, output_dir
from .manifest import write_manifest


def cmd_show_config(config: dict, args: argparse.Namespace) -> int:
    print(json.dumps({k: v for k, v in config.items()}, indent=2, default=str))
    print(f"config_hash: {config_hash(config)}")
    return 0


def cmd_manifest(config: dict, args: argparse.Namespace) -> int:
    path = write_manifest(output_dir(config, "manifests"), config, "manifest")
    print(path)
    return 0


COMMANDS = {
    "show-config": (cmd_show_config, "Print the resolved study config and its hash"),
    "manifest": (cmd_manifest, "Write a run manifest (git, config hash, packages)"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hcmv", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"hcmv {__version__}")
    parser.add_argument("--config", default=None, help="Study YAML (default: configs/study.yaml)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value with a dotted key, e.g. --set cv.outer_repeats=1",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    handler, _ = COMMANDS[args.command]
    return handler(config, args)


if __name__ == "__main__":
    sys.exit(main())

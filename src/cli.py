"""Command-line entry point for EEG workflows."""

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "public-esn": ("models.tune_public_esn", "Tune an ESN on training sessions and evaluate a separate session."),
    "public-fisher": ("analysis.compute_public_fisher", "Run public Fisher analysis with optional ESN evaluation."),
    "bel": ("models.decode_bel_words", "Decode marked BEL EGI recordings with a sparse ridge ESN."),
    "preprocess": ("data.preprocess_biosemi_bdf", "Convert BioSemi BDF recordings to EEG FIF epochs."),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="neurodecoder",
        description=__doc__,
        epilog="\n".join(f"  {name:18} {help_text}" for name, (_, help_text) in COMMANDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS)
    if not args:
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    module = import_module(f".{COMMANDS[command][0]}", package=__package__)
    module.main(args[1:])


if __name__ == "__main__":
    main()

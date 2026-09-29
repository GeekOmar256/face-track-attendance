"""Start the Face Track desktop window (Tkinter).

    python scripts/run_gui.py

This is the local alternative to the web interface. Prefer scripts/run_web.py
on the Raspberry Pi: this window needs a display, which on a headless board
means X11 forwarding over SSH and a much slower preview.
"""

from __future__ import annotations

import argparse

from _bootstrap import banner  # noqa: F401  (puts the project root on sys.path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=("haar", "yunet"), default="haar",
                        help="detector selected when the window opens (default: %(default)s)")
    args = parser.parse_args()

    try:
        import tkinter  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\ntkinter is not installed.\n"
            "  Raspberry Pi / Debian:  sudo apt install python3-tk\n"
            "  or use the web interface instead:  python scripts/run_web.py\n"
        )

    from facetrack.gui import main as gui_main
    return gui_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Start the Face Track web interface.

    python scripts/run_web.py
    python scripts/run_web.py --port 8080

Then open the address it prints. On the Raspberry Pi this needs no display and
no X11 forwarding: the page is served over the local network, which is how the
FYP1 plan describes the dashboard reaching the board.
"""

from __future__ import annotations

import argparse

from _bootstrap import banner  # noqa: F401  (puts the project root on sys.path)
from facetrack.webapp import serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0",
                        help="interface to bind, 127.0.0.1 keeps it on this machine "
                             "(default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000,
                        help="port to listen on (default: %(default)s)")
    args = parser.parse_args()
    return serve(host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())

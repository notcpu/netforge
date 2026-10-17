#!/usr/bin/env python
"""
Graphical neural network trainer entry point.

    python netforge.py                 open the GUI
    python netforge.py solve 7b        architecture for a parameter target
    python netforge.py train --solve 5m
    python netforge.py memory --config my.run.json

Multiprocessing freeze_support is required on Windows before anything touches
tkinter, or a console-launched script can re-enter itself.
"""

import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) == 1:
        from netforge.gui import launch
        launch()
        return 0
    from netforge.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
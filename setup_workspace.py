"""
Setup Script.

Use this script to cerate the user.email and user.name entries necessary in .gitconfig
This script will also set up a virtual environment from the global distribution and
install all needed dependencies. (This is unfortunately necessary as the global distribution
is not on the path for new terminals)

NOTE: this script is intended to be run on the level 3 windows computers. Having said that it
should work on your personal computers (regardless of OS) but has not been tested on an OS other
than Windows.
"""
import sys
import logging
import platform
from subprocess import run
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s : %(message)s")

logging.info("Starting script...")

logging.info("Detecting platform...")
SYSTEM = platform.system()
if not SYSTEM:
    logging.error("Couldn't determine the operating system")
    sys.exit(1)
logging.info("Found operating system: %s", 'MacOS' if SYSTEM == 'Darwin' else SYSTEM)

logging.info("GitHub information required")
name = input("Enter your name: ")
email = input("Enter the email address registered with GitHub: ")
run(['git', 'config', '--global', 'user.email', email], check=False, stdout=sys.stdout, stderr=sys.stderr)
run(['git', 'config', '--global', 'user.name', name], check=False, stdout=sys.stdout, stderr=sys.stderr)


logging.info("Creating (and updating) new python virtual environment...")
run([sys.executable, "-m", "venv", "venv"], check=False, stdout=sys.stdout, stderr=sys.stderr)

VENV_INTERPRETER = 'venv/Scripts/python.exe' if SYSTEM == "Windows" else r"venv\bin\python"
logging.info("Using virtual env interpreter: %s", VENV_INTERPRETER)
run([VENV_INTERPRETER, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'],
    check=False, stdout=sys.stdout, stderr=sys.stderr)

logging.info("Installing dependencies...")
run([VENV_INTERPRETER, '-m', 'pip', 'install', '-e', '.'], check=False, stdout=sys.stdout, stderr=sys.stderr)

logging.info("Setup complete!")
logging.warning("Please re-launch the terminal for changes to take effect.")

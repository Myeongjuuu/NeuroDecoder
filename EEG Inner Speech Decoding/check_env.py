"""Sanity-check script: verify that core dependencies are importable and report versions."""
import importlib
import sys

_REQUIRED = [
    "mne",
    "numpy",
    "scipy",
    "sklearn",
    "matplotlib",
    "pandas",
]

def main():
    print(f"Python {sys.version}")
    all_ok = True
    for pkg in _REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {pkg:<12} {version}")
        except ImportError:
            print(f"  {pkg:<12} NOT FOUND")
            all_ok = False
    if all_ok:
        print("\nAll required packages are available.")
    else:
        print("\nSome packages are missing. Run: pip install -e '.[dev]'")

if __name__ == "__main__":
    main()

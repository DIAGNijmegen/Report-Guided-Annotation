# Notes for developers

## Push release to PyPI
1. Increase version in setup.py, and set below
2. Build: `python -m build`
3. Test package distribution: `python -m twine upload --repository testpypi dist/*0.2.6*`
4. Distribute package to PyPI: `python -m twine upload dist/*0.2.6*`

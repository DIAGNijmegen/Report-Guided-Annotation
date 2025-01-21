import setuptools

if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        version='0.3.3',  # note: also push to PyPI
        author_email='Joeran.Bosma@radboudumc.nl',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/DIAGNijmegen/Report-Guided-Annotation',
        project_urls={
            "Bug Tracker": "https://github.com/DIAGNijmegen/Report-Guided-Annotation/issues"
        },
        license='MIT',
        packages=['report_guided_annotation'],
    )

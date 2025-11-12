from setuptools import setup, find_packages


setup(
    name="radas",
    packages=find_packages(),
    install_requires=[
        # core
        "ray",
        "pyarrow",
        "pydantic",
        "torch",
        "pandas",
        "seaborn",
        "scipy",
        "tqdm",
        #
        # for notebooks
        "ipykernel",
        "ipywidgets",
        #
        # for parallel coordinates plot
        "plotly",
        "nbformat",
    ],
    python_requires=">3.8,<3.13",
)

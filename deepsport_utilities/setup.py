from setuptools import setup, find_packages

setup(
    name='deepsport_utilities',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@uclouvain.be",
    url="https://gitlab.com/deepsport/deepsport_utilities",
    licence="LGPL",
    python_requires='>=3.8',
    # 3.8 required for
    #    - PEP 572 – Assignment Expressions (:=)
    #    - `functools.cached_property` (although `mlworkflow.lazyproperty` could be used for older python versions)
    # 3.7 required for
    #    - EvalAI ... but we don't care anymore, do we?
    description="",
    version='4.10.1',
    packages=find_packages(),
    install_requires=[
        "numpy", # was >=1.20", ... but I don't understand why
        "scipy",
        "opencv-python",
        "imageio",
        "m3u8",
        "requests",
        "calib3d>=2.10.1",
        "mlworkflow>=0.6.0",
        "shapely",
        "scikit-image",
        "aleatorpy"
    ],
    extras_require={
        'dev': [
            "pdoc3",
            "pytest",
            "twine",
        ]
    }
)

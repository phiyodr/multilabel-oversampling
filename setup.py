import setuptools
from multilabel_oversampling import __version__
print(__version__)

with open("README.md", "r") as file:
    long_description = file.read()

with open("requirements.txt") as file:
    required = file.read().splitlines()
    
setuptools.setup(
    name="multilabel-oversampling",
    version=__version__, 
    author="Philipp J. Rösch",
    author_email="phiyodr@gmail.com",
    description="Multilabel Oversampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phiyodr/multilabel-oversampling",
    classifiers=[
    	"Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.csv", "*.json"]},
    install_requires=[
            "numpy",
            "scikit-learn",
            "pandas",
            "seaborn",
            "tqdm",
            "matplotlib"
        ]
    )

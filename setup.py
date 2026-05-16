from setuptools import find_packages, setup

setup(
    name="eco-smart-classifier",
    version="1.0.0",
    description="Waste classification and resale value estimation pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)

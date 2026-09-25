from setuptools import find_packages, setup


def get_requirements(file_path):
    with open(file_path, "r") as file:
        requirements = [
            requirement.strip()
            for requirement in file.readlines()
            if requirement.strip()
        ]

    if "-e ." in requirements:
        requirements.remove("-e .")

    return requirements


setup(
    name="amazon-ml-entity-resolution",
    version="0.0.1",
    author="Your Name",
    author_email="your-email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
)
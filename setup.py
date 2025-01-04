from setuptools import setup, find_packages

setup(
    name="neural_control_system",
    version="2.0.0",
    packages=find_packages(include=["sources", "sources.*", "utils", "utils.*"]),
    install_requires=[
        "scikit-learn==1.4.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.8.4"
    ],
    entry_points={
        'console_scripts': [
            'neural_from_scratch=main:main',  # If you want to create a command-line interface
        ],
    },
    description="A neural control package built from scratch",
    author="Razani-Ali",
    author_email="ali.razani2001@protonmail.com",
    url="https://github.com/Razani-Ali/neural_control_system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

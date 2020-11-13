#!/usr/bin/env python
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ShinySilver",  # Replace with your own username
    version="0.1.0",
    author="Louis-Marie NICOLAS & Corentin GAILLARD",
    author_email="author@example.com",
    description="Fake News Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShinySilver/SIIA-FakeNewsDetection",
    packages=['ias-fakedetection'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

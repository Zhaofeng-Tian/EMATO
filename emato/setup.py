from setuptools import setup, find_packages

setup(
    name="emato",  # Replace with your project name
    version="0.1.0",  # Replace with your project version
    description="A highway simulator with trajectory generation and lane changing",  # Brief description of your project
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/yourusername/emato",  # Replace with your project's URL
    packages=find_packages(where="."),  # Automatically find packages in the directory
    install_requires=[
        # List your project's dependencies here
        # For example:
        # 'numpy',
        # 'torch',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose the appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
)

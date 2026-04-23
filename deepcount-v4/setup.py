from setuptools import setup, find_packages

setup(
    name="deepcount",
    version="0.1.0",
    description="Hierarchical RL agent that learns to card-count at blackjack",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
    ],
)

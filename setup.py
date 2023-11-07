# Adapted from https://github.com/McGill-NLP/bias-bench

from setuptools import setup

setup(
    name="LLM-Pruning-And-Fairness",
    version="0.1.0",
    description="Can Pruning Language Models Reduce Their Bias?",
    url="https://github.com/bhattaraiprayag/LLM-Pruning-And-Fairness",
    packages=["LLM-Pruning-And-Fairness"],
    install_requires=[
        "torch==1.10.2",
        "transformers==4.16.2",
        "scipy==1.7.3",
        "scikit-learn==1.0.2",
        "nltk==3.7.0",
        "datasets==1.18.3",
        "accelerate==0.5.1",
    ],
    include_package_data=True,
    zip_safe=False,
)

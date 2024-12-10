from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ssdllm",
    version="0.1.0",
    author="LLM as Dataset Analyst",
    author_email="",
    description="SSD-LLM: Subpopulation Structure Discovery via Large Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/llm-as-dataset-analyst/SSDLLM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=required,
    include_package_data=True,
)
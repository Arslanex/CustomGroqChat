from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="customgroqchat",
    version="0.1.0",
    author="Enes Arslan",
    author_email="enes@vidinsight.com.tr",
    description="Python client for the Groq Cloud API with intelligent rate limiting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arslanex/CustomGroqChat",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "tiktoken>=0.4.0",
        "asyncio>=3.4.3",
    ],
    keywords="groq, api, nlp, ai, language model, rate limiting, queue",
    project_urls={
        "Documentation": "https://github.com/Arslanex/CustomGroqChat/docs",
        "Source": "https://github.com/Arslanex/CustomGroqChat",
        "Issue Tracker": "https://github.com/Arslanex/CustomGroqChat/issues",
    },
)

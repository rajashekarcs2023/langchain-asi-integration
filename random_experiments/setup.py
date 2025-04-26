"""Setup for langchain-asi package."""
from setuptools import setup, find_packages

setup(
    name="langchain-asi",
    version="0.1.0",
    description="LangChain integrations for ASI AI.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Rajashekar Vennavelli",
    author_email="rajashekar.vennavelli@fetch.ai",
    url="https://github.com/rajashekarcs2023/langchain-asi-integration",  # Updated GitHub repo URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "langchain-core>=0.1.0",
        "httpx>=0.24.1",
        "requests>=2.31.0",
        "typing-extensions>=4.7.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "examples": [
            "langchain>=0.3.0",
            "langgraph>=0.0.26",
        ],
    },
    python_requires=">=3.8.1",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
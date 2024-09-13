from setuptools import setup, find_packages

setup(
    name='translation',
    version='0.1.0',
    packages=find_packages(include=['translation', 'translation.*']),
    install_requires=[
        "transformers",
        "torch",
        "datasets",
        "peft",
        "bitsandbytes",
        "numpy<2.0.0",
        "ipykernel",
        "ipywidgets",
        "jsonlines", 
    ],
        extras_require={ # pip install -e .[evaluation]
        'evaluation': [
            "rouge", 
        ]
    }
)
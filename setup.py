import pathlib

import setuptools

_here = pathlib.Path(__file__).resolve().parent

name = "mbrlax"
author = "Aidan Scannell"
author_email = "scannell.aidan@gmail.com"
description = "Bayesian model-based reinforcement learning in PyTorch."

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/aidanscannell/" + name

license = "Apache-2.0"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]
keywords = [
    "model-based-reinforcement-learning",
    "bayesian-deep-learning",
    "deep-learning",
    "machine-learning",
    "bayesian-inference",
    "planning",
]

python_requires = "~=3.7"

install_requires = [
    "torch",
    "functorch",  # needed for vmap
    "laplace-torch",
    "gpytorch",
    # "laplace",
    "numpy",
    "matplotlib",
    "gymnasium",
    "mujoco",
    "torchtyping",
    "pytorch_lightning",
]
extras_require = {
    # "experiments": ["hydra-core", "palettable", "tikzplotlib"],
    # "examples": ["jupyter", "hydra-core"],
    # "dev": ["black[jupyter]", "pre-commit", "pyright", "isort", "pyflakes", "pytest"],
    "dev": ["black", "pyright", "isort", "pyflakes", "pytest"],
    "examples": [
        # "dm_control",
        "tikzplotlib",
        "bsuite",
        "ipython",
        "seaborn",
        "hydra-core",
        "wandb",
    ],
}

setuptools.setup(
    name=name,
    version="0.1.0",
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    keywords=keywords,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    # packages=setuptools.find_packages(include=["examples"]),
    # packages=setuptools.find_packages(exclude=["examples","tests"]),
    # packages=setuptools.find_packages(exclude=["tests"]),
    # packages=setuptools.find_packages(include=["examples"]),
    # packages=setuptools.find_packages(include=["experiments", "experiments.*"]),
    # packages=setuptools.find_packages(exclude=["tests"]),
    # packages=setuptools.find_packages(),
    # packages=setuptools.find_packages(exclude=["paper"]),
    packages=setuptools.find_namespace_packages(),
    # packages=setuptools.find_namespace_packages(where="mbrlax"),
)

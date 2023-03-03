import os
import pathlib
from setuptools import setup, find_packages

pkg_name = "pirl"
package_dirs = {pkg_name: pkg_name}

top_dir = pathlib.Path(__file__).parent

with open(top_dir.joinpath("requirements.txt"), "r") as req_file:
    requirements = [line.strip() for line in req_file.readlines()]

with open(top_dir.joinpath("optional-requirements.txt"), "r") as req_file:
    extras = [line.strip() for line in req_file.readlines()]

with open("LICENSE", "r") as f:
    license_text = f.read()

setup(name="pirl",
      version="0.1",
      author="Can Bogoclu, Robert Vosshall",
      description="Probabilistic inference for reinforcement learning",
      license=license_text,
      package_dir=package_dirs,
      packages=[pkg_name + "." + p for p in find_packages(where=pkg_name)],
      py_modules=[pkg_name + ".__init__"],
      install_requires=requirements,
      extras_require={"full": extras},
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: Microsoft :: Windows",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python :: 3 :: Only"
          "Programming Language :: Python :: 3.8",
      ],
      keywords=["reinforcement learning",
                "uncertainty propagation",
                "machine learning",
                "model predictive control",
                "optimal control"]
      )

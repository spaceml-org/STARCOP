from setuptools import setup, find_packages
import codecs
import os.path

with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name="starcop",
      version=get_version("starcop/__init__.py"),
      author="Vit Rutzicka, Gonzalo Mateo-Garcia",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(".", exclude=["tests"]),
       package_data={
        "starcop" : [
                    "models/ch4.lut", 
                    "models/ch4.hdr",
                    "config.yaml"]
       },
      description="Semantic segmentation of methane plumes with hyperspectral machine learning models",
      install_requires=parse_requirements_file("requirements_package.txt"),
      keywords=["ch4", "methane", "remote sensing"],
)

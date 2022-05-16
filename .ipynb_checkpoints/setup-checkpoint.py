import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='domino',
    version='0.02',
    author='Josh Dorrington',
    author_email='joshua.dorrington@kit.edu',
    description='A package for compositing atmospheric datasets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/joshdorrington/domino',
    license='bsd-3-clause',
    packages=setuptools.find_packages()
)
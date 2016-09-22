from setuptools import setup, find_packages

import dslib


setup(
    # Info
    name='dslib',
    version=dslib.__version__,

    # Author
    author='Yurii Shevhcuk',
    author_email='mail@itdxer.com',

    # Package
    packages=find_packages(),
)

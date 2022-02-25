"""
Setup installation file for the skaggs package.
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='skaggs',
    version='0.1.0',
    description='skaggs - Neural Recording Analysis Tools',
    long_description=long_description,
    url='https://github.com/jdmonaco/skaggs',
    author='Joseph Monaco',
    author_email='self@joemona.co',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='neuroscience data analysis recording navigation spikes',
    packages=['skaggs'],
)

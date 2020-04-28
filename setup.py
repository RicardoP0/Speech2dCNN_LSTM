#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='Speech2dCNN_LSTM',
      version='0.0.1',
      description='A pytorch implementation of Speech emotion recognition using deep 1D & 2D CNN LSTM networks',
      author='',
      author_email='',
      url='https://github.com/RicardoP0/Speech2dCNN_LSTM.git',
      install_requires=[
            'pytorch-lightning'
      ],
      packages=find_packages()
      )


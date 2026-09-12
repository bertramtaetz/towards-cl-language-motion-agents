#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


if __name__ == '__main__':
    requirements_path = 'requirements.txt'
    if sys.version_info[0] < 3:
        requirements_path = 'requirements_py2.txt'
    with open(requirements_path) as f:
        install_reqs = f.read().splitlines()

    # Ensure Python 3.12 compatibility: gensim 3.x is not supported.
    # (requirements.txt is the source of truth, but we keep this guard here
    # so that packaging metadata is correct even if an old requirements file is used.)
    install_reqs = [
        ('gensim>=4.0' if r.strip().startswith('gensim~=3.') else r)
        for r in install_reqs
    ]

    setup(name='nlg-eval',
          version='2.4.1',
          description="Wrapper for multiple NLG evaluation methods and metrics.",
          author='Shikhar Sharma, Hannes Schulz, Justin Harris',
          author_email='shikhar.sharma@microsoft.com, hannes.schulz@microsoft.com, justin.harris@microsoft.com',
          url='https://github.com/Maluuba/nlg-eval',
          packages=find_packages(),
          include_package_data=True,
          scripts=[],
          install_requires=install_reqs,
    )

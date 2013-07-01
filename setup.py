try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'EOLDAS ng',
    'author': 'J Gomez-Dans',
    'url': 'http://github.com/jgomezdans/eoldas_ng/',
    'author_email': 'j.gomez-dans@ucl.ac.uk',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['eoldas_ng'],
    'scripts': [],
    'name': 'eoldas_ng'
}

setup(**config)

from setuptools import setup

setup(
    name='RME',
    version='0.0.1',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn'
    ],
    packages=['preprocessor','explainer','plots'],
    url='',
    license='',
    author='Mateusz Kobylka',
    author_email='kobylkam95@gmail.com',
    description='Recurrent Memory Explainer'
)

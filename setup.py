from setuptools import setup, find_packages

setup(
    name='RME',
    version='0.0.1',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn'
    ],
    packages= find_packages(exclude=['tests']),
    py_modules=['preprocessor','plots','explainer']
    include_package_data=True,
    url='https://github.com/kobylkam/RME',
    license='',
    author='Mateusz Kobylka',
    author_email='kobylkam95@gmail.com',
    description='Recurrent Memory Explainer'
)

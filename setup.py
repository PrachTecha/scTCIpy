from setuptools import setup, find_packages

setup(
    name='scTCIpy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'anndata',
        'scanpy',
        'scipy',
        'matplotlib',
        'tqdm'
    ],
    author='Prach Techameena',
    author_email='prach.techa@gmail.com',
    description='A package for calculating the cell transition indices to identify the transitioning cells.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PrachTecha/scTCIpy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',

)
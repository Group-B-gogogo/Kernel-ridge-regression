from setuptools import setup, find_packages

setup(
    name='kernel_ridge',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A implementation of Kernel Ridge Regression',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/kernel-ridge-regression',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().splitlines(),
)

from distutils.core import setup
setup(
    name = 'snoprop',
    packages = ['snoprop'],
    version = '1.0',
    license='Other/Proprietary License',
    description = 'Solver for Nonlinear Optical Propagation',
    author = 'John Ryan Peterson',
    author_email = 'jrpeterson.physics@gmail.com',
    url = 'https://github.com/USNavalResearchLaboratory/SNOPROP/releases/latest',
    download_url = 'https://github.com/USNavalResearchLaboratory/SNOPROP/archive/v1.0.tar.gz',
    keywords = ['simulation','lasers'],
    install_requires=[
        'numpy',
        'scipy',
        'numba>=0.42',
    ],
    python_requires='>=3.5, <4',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: OS Independent',
    ],
    data_files = [('', ['LICENSE.txt'])]
)

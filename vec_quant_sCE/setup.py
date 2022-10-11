from pathlib import Path
from setuptools import find_packages, setup


install_requires = Path('requirements.txt').read_text().splitlines()

setup(
    name='vec-quant-sCE',
    packages=find_packages(where='vec_quant_sCE'),
    package_dir={'': 'vec_quant_sCE'},
    version='0.1.0',
    description=(
        'Vector quantization for synthetic contrast enhancement'
    ),
    author='Mark Pinnock',
    license='MIT',
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'train=vec_quant_sCE.training.cli:main',
            'predict=cxr_tube.cli.inference:main',
        ],
    },
)
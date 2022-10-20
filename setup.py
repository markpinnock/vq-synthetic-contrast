from setuptools import find_packages, setup


setup(
    name='vec-quant-sCE',
    packages=find_packages(),
    version='0.1.0',
    description=(
        'Vector quantization for synthetic contrast enhancement'
    ),
    author='Mark Pinnock',
    license='MIT',
    entry_points={
        'console_scripts': [
            'train=vec_quant_sCE.cli.training:main',
            'predict=vec_quant_sCE.cli.inference:main',
        ],
    },
)
from setuptools import find_namespace_packages, setup


setup(
    name='vq-sce',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    version='0.1.0',
    description=(
        'Vector quantization for synthetic contrast enhancement'
    ),
    author='Mark Pinnock',
    license='MIT',
    entry_points={
        'console_scripts': [
            'train=vq_sce.cli.training:main',
            'predict=vq_sce.cli.inference:main',
        ],
    },
)
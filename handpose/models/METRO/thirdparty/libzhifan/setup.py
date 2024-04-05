from setuptools import find_packages, setup

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

setup(
    name='libzhifan',
    author="Zhifan Zhu",
    author_email="zhifan.zhu@bristol.ac.uk",
    packages=find_packages(exclude=("tests",)),
    # package_dir = {'libzhifan': 'libzhifan'},
    install_requires=[
        'numpy>=1.16.3',
        'matplotlib>=2.1.0',
        'pillow>=6.0.0',
        'trimesh>=3.10.2',
        # 'pytorch3d==0.6.2', suggested
    ],
    version='0.1',
)

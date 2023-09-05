import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='cppcuda_tutorial',
    version='1.0',
    author='chengxiao',
    author_email='vegtsunami@gmail.com',
    description='cppcuda tutorial',
    long_description='cppcuda tutorial',
    ext_modules=[
        CUDAExtension(
            name='cppcuda_tutorial',
            sources=sources,
            include_dirs=include_dirs,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
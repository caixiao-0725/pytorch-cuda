from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='cppcuda_tutorial',
    version='1.0',
    author='chengxiao',
    author_email='vegtsunami@gmail.com',
    description='cppcuda tutorial',
    long_description='cppcuda tutorial',
    ext_modules=[
        CppExtension(
            name='cppcuda_tutorial',
            sources=['interpolation.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
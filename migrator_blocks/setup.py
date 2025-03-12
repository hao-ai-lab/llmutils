from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='migration_ops',
    ext_modules=[
        CUDAExtension('migration_ops', ['migration_ops.cc'],
        extra_compile_args={'cxx': ['-g'],
                          'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 

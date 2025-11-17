from setuptools import setup
import cmake_build_extension

setup(
    name="cupgs",
    version="0.1.0",
    author="Filippo Luca Ferretti",
    author_email="filippoluca.ferretti@outlook.com",
    description="CUDA-based Projected Gauss-Seidel solver with multi-GPU support",
    long_description="",
    package_dir={"": "python"},
    cmdclass=dict(build_ext=cmake_build_extension.BuildExtension),
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="BuildAndInstall",
            install_prefix="cupgs",
        )
    ],
)

from setuptools import setup, Extension
import os

def build_extension():

    try:
        from Cython.Build import cythonize
        use_cython = True
        ext = ".pyx"
    except ImportError, e:
        use_cython = False
        ext = ".c"


    l2s = Extension(
        "l2sum._learn",
        sources=[
            os.path.join("l2sum", "_learn{}".format(ext)),
        ],
        #extra_compile_args=["-std=c11" ],
        #extra_compile_args=["-O9", "-std=c11" ],
        #extra_link_args=["-O9"],
        language="c")

    if use_cython is True:
        return cythonize([l2s], gdb_debug=False)
    else:
        return [l2s]

setup(
    name="l2sum",
    version="0.0.1",
    author="Chris Kedzie",
    author_email="kedzie@cs.columbia.edu",
    description="Learning 2 search for summarization.",
#    license="",
#    keywords="",
#    url="",
    packages=['l2sum'],
#    long_description="",
#    classifiers=[],
    ext_modules=build_extension(),
)

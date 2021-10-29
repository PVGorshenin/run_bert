import re
import setuptools

(__version__,) = re.findall("__version__.*\s*=\s*[']([^']+)[']",
                            open('run_bert/__init__.py').read())

setuptools.setup(
    name="run_bert",
    version=__version__,
    packages=setuptools.find_packages(),
    python_requires="<3.9.0",
    install_requires=[
        "numpy==1.18.0",
        "tqdm==4.41.1",
        "transformers==2.5.1",
        "torch==1.5.1"
    ],
)

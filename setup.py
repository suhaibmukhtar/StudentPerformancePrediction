#using this we can deploy our project in PYPI and someone can install it using pip
from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path:str)->List[str]:
    """"
    This function takes the path of requirements.txt as input and will load/install the libraries from their
    i.e. this function returns the list of requirements from the file_path mentioned.
    """
    requirements=[]
    with open(file_path,'r') as file_obj:
        requirements=file_obj.readlines()
    requirements=[req.replace("\n","").strip() for req in requirements]
    HYPHEN_E_DOT="-e ."
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT) 
    return requirements
    
setup(
    name="mlproject",
    version="0.0.1",
    author='SuhaibMukhtar',
    author_email="suhaibmukhtar2@gmail.com",
    packages=find_packages(),
    # install_requires=['numpy','pandas']here we can mention libraries we want to install(dependency)
    #or above one not preferrable
    install_requires=get_requirements('requirements.txt')
)
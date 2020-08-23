from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='basicStats',
      version='0.0.0',
      description='Extract Basic Statistics from Data',
      long_description=readme(),
      classifiers=[],
      url='',
      author='AnnMelanieHuang',
      author_email='ann.melanie.huang@gmail.com',
      install_requires=['numpy', 'scipy' ,'pandas'],
      packages=['basicStats'],
      include_package_data=True,
      zip_safe=False)

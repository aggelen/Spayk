from setuptools import setup

setup(
    name='Spayk',
    version='0.1.1',    
    description='SPAYK: An environment for spiking neural network simulation',
    url='https://github.com/aggelen/Spayk',
    author='Aykut Görkem Gelen',
    author_email='aggelen@outlook.com',
    license='BSD 3-clause',
    packages=['spayk'],
    install_requires=['matplotlib==3.5.0',
                      'numpy==1.21.2',
                      'seaborn==0.11.2',
                      'tqdm==4.62.3',    
                      'torchvision==0.10.0',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)



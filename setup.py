from setuptools import setup, find_packages

setup(
    name='AAAKD',
    version='0.1.0',
    description='Attention-Aware Adaptive Knowledge Distillation for Vision Transformers',
    author='Anonymous',
    author_email='anonymous@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.2',
        'torchvision>=0.16.2',
        'timm>=0.9.12',
        'numpy>=1.26.3',
        'pillow>=10.2.0',
        'tqdm>=4.67.1',
        'pandas>=2.2.3',
        'tensorboard>=2.18.0',
        'wandb>=0.16.1',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)

from setuptools import setup, find_packages
import os

setup(
    name="decalib",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'decalib': [
            '../data/*',
            '../configs/release_version/*',
            '../TestSamples/examples/*',
            '../TestSamples/AFLW2000/*',
            '../TestSamples/exp/*',
            '../TestSamples/teaser/*',
        ],
        '': ['data/*', 'configs/release_version/*']
    },
    data_files=[
        ('deca_data', [
            'data/deca_model.tar',
            'data/female_model.pkl',
            'data/male_model.pkl',
            'data/generic_model.pkl',
            'data/fixed_displacement_256.npy',
            'data/FLAME_albedo_from_BFM.npz',
            'data/head_template.obj',
            'data/landmark_embedding.npy',
            'data/mean_texture.jpg',
            'data/resnet50_ft_weight.pkl',
            'data/texture_data_256.npy',
            'data/uv_face_eye_mask.png',
            'data/uv_face_mask.png',
        ]),
        ('deca_configs', [
                'configs/release_version/deca_coarse.yml',
            'configs/release_version/deca_detail.yml', 
            'configs/release_version/deca_pretrain.yml',
        ])
    ],
    install_requires=[],
    python_requires='>=3.7',
    description="Detailed Expression Capture and Animation",
    long_description=open('README.md').read() if os.path.exists('README.md') else "",
    long_description_content_type="text/markdown",
    author="DECA authors",
    url="https://github.com/YadiraF/DECA",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)


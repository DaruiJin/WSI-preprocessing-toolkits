# WSI-preprocessing-toolkits

The WSI Preprocessing Toolkit is a user-friendly library designed to streamline the preprocessing of whole slide images (WSIs) and thus simplify the initial steps of histopathological deep learning projects. This toolkit empowers you to efficiently preprocess whole slide images (WSIs) by dividing slide images into tiles at various magnifications, while also generating thumbnails for the mask of the effective region and enabling seamless stitching.

Optimizing CPU utilization is crucial for improving the speed of tiling. In our codes, [concurrent library](https://docs.python.org/3/library/concurrent.futures.html) is also used to distribute jobs across multiple CPU cores, leading to improved processing efficiency and reduced execution times.

It should also be mentioned that this project is inspired by the collaboration of related codebases including [CLAM](https://github.com/mahmoodlab/CLAM), [wsi-preprocessing](https://github.com/lucasrla/wsi-preprocessing), and [End-to-end WSI Processing Pipeline](https://github.com/KatherLab/end2end-WSI-preprocessing).

## Pre-requisites:
Use requirements.txt to install required packages.
```
pip install -r requirements.txt
```
## How to use:

## License


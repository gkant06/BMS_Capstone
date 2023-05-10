# BMS_Capstone

In the field of pharmaceutical development, image analysis plays a crucial role in various applications such as microscopy and X-ray diffraction. However, traditional imaging approaches often rely on manual or semi-manual processes, which are time-consuming, prone to errors, and require specialized training. To address these challenges, machine learning techniques have emerged as a powerful tool to automate image analysis and interpretation. By leveraging the capabilities of machine learning algorithms, the processing of images can be streamlined, leading to faster results and reduced human error. Moreover, machine learning enables the development of reusable components that can be applied to diverse datasets and analysis tasks. Nevertheless, one of the key challenges in applying machine learning to image analysis is the lack of labeled or annotated data, as many existing techniques are based on supervised learning paradigms. In this context, unsupervised learning approaches become essential for tackling image analysis problems where ground truth or labeled data is limited.

One approach to achieve this is by generating pseudo-labels using machine learning pipelines. By applying machine learning techniques, meaningful cluster labels can be assigned to images, such as distinguishing between ill-formed and well-formed protein droplets or identifying different conformations of HS-AFM molecules. This process reduces the need for manual labeling of new data, saving time and effort. Furthermore, the generated labels can be utilized for various tasks including classification, object recognition, and object tracking, enabling the reuse of components and pipelines across different datasets. This project highlights the value of employing machine learning techniques to generate pseudo-labels and demonstrates their potential to enhance efficiency and productivity in image analysis workflows.

1. Datasets
- Protein crystalization images
- High-speed Atomic Force Microscopy (HS-AFM) videos


2. Workflow
![Workflow](https://github.com/gkant06/BMS_Capstone/assets/112508461/5f306366-9124-4370-be5d-e05fc719d3c4)

- BMS_capstone/classification is the classification pipeline for preprocessing of protein crystalization images
- BMS_capstone/segmentation is the preprocessing pipeline for HS-AFM dataset
- BMS_capstone/feature_extraction is the pipeline for extracting latent features of data and performing clustering. Here we only use K-means method and the number of clusters need to be determined by users.


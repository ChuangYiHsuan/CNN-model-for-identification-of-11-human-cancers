### CNN model for identification of 11 human cancers  

The Convolutional Neural Network (CNN) serves as a powerful model for medical image identification. This project aims to transform protein-protein interaction data into 2D image coordinates (i.e., X-Y axis) using spectral clustering techniques, including Laplacian matrix, eigenvalues, and eigenvectors. The protein-protein interaction network is mapped in to X-Y axis, and Next-Generation Sequencing (NGS) gene expression values determine the "color intensity" of the image (e.g., high-expression genes appear in red, while low-expression genes appear in green). This transformation generated 6,136 distinct tumor 2D images for 11 types of cancer. These images are then input into a CNN model for the classification of 11 cancer types and one normal type.

### The main scripts for our CNN model include:  
**1. Step1_GenerateNpyFile.py**  
**2. Step2_Build_CNN_model.py**  
**3. Step3_IndependentValidation.py**  

In the Dataset Preparation section below, we introduce the model input. The provided code snippets highlight crucial steps in CNN creation; for the complete code, please download above code files or refer to our published paper.

Our Python version is 3.7 and tensorflow version is 2.4.1.  
**For more detailed information, please refer to our published paper:**  
**https://pubmed.ncbi.nlm.nih.gov/34667236/**  
**(Sci Rep. 2021 Oct 19;11(1):20691)**
  
  
### Dataset Preparation ([The example files](https://github.com/ChuangYiHsuan/CNN-model-for-identification-of-11-human-cancers/tree/main/input_examples))  
  
The human protein-protein interactions (PPIs) were collected, including 16,433 human proteins and 181,868 PPIs. These PPIs were transformed into a Laplacian matrix (i.e., an adjacency and diagonal matrix) to be mapped onto a X-Y axis (black and white 2D image). Subsequently, we gathered numeric Next-Generation Sequencing (NGS) data for 11 cancer types (5,528 tumors and 608 normal tissues). The data values were mapped into images, with colors representing high or low numerical values. These colored images served as inputs for the CNN model to identify distinct cancers.
  
### Step1_GenerateNpyFile.py  
In the first step, our aim is to read files (which can be considered as colored images) containing gene expression values within a 100x100-sized matrix of each sample. Subsequently, we generate *.npy files to serve as inputs for the CNN model.  
```python
# We shuffle the order of image files
# to prevent the CNN model from classifying samples based on file order.
seed = str(time.year) + str(time.month) + str(time.day) +
str(time.hour) + str(time.minute) + str(time.microsecond)
random.seed(seed)
random.shuffle(dirs)
.
.
.
# Subsequently, the program generates 3 files:
# (1) a single file that includes all training images (in our case, totaling 1,228 images),
# (2) a file recording ground truth labels, and (3) a file recording sample IDs.
np.save(output_file_name + ".npy", x_samples) # npy file
np.save(output_file_name + "_label.npy", y_labels) # truth labels of samples
np.save(output_file_name + "_title.npy", sample_titles) # unique IDs of samples
```

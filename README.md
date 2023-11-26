### CNN model for identification of 11 human cancers  

The Convolutional Neural Network (CNN) serves as a powerful model for medical image identification. This project aims to transform protein-protein interaction data into 2D image coordinates (i.e., X-Y axis) using spectral clustering techniques, including Laplacian matrix, eigenvalues, and eigenvectors. The protein-protein interaction network is mapped into X-Y axis, and Next-Generation Sequencing (NGS) gene expression values determine the "color intensity" of the image (e.g., high-expression genes appear in red, while low-expression genes appear in green). This transformation generated 6,136 distinct tumor 2D images for 11 types of cancer. The images are then input into a CNN model for the classification of 11 cancer types and one normal type.

### The main programs for our CNN model include:  
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
...
# Subsequently, the program generates 3 files:
# (1) a single file that includes all training images (in our case, totaling 1,228 images),
# (2) a file recording ground truth labels, and (3) a file recording sample IDs.
np.save(output_file_name + ".npy", x_samples) # a npy file resprent all training images  
np.save(output_file_name + "_label.npy", y_labels) # truth labels of images
np.save(output_file_name + "_title.npy", sample_titles) # unique IDs of images
```

### Step2_Build_CNN_model.py  
In second step, we build CNN model by Sequential() from keras.models.  
```python
# Build CNN model (Keras)
model = Sequential()
# Conv 1: filter 5*5, kernel number=64, input size 100*100
model.add(Conv2D(filters = 64, kernel_size = (5, 5), input_shape = (100, 100, 1)))
model.add(B_nor(axis = 2, epsilon = 1e-5)) # batch normalization, accelerate convergence, avoid gradient vanishing/exploding
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same")) # same: padding, "same" size of input
# Conv 2
model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(B_nor(axis = 2, epsilon = 1e-5))
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
# Conv 3
model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(B_nor(axis = 2, epsilon = 1e-5))
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
# Fully connect
model.add(Flatten())
# Hidden layers, three layers with 1000, 600 and 80 neuron nodes
model.add(Dense(1000, activation = "relu"))
model.add(Dense(600, activation = "relu"))
model.add(Dense(80, activation = "relu"))
model.add(Dense(12, activation = "softmax")) # 12 label classes, 11 cancers + 1 normal type
...
# We utilize the Matthews Correlation Coefficient (MCC) as our classification metric.
def MCC(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
```
### Step3_IndependentValidation.py  
In this step, we load a previously trained model and assess its performance using independent testing sets for validation.
```python
# load weight from previous training
try:
    model.load_weights(input_save_model)
    print("Successfully loading previous training weights.")
except:
    print("Failed to load previous data, training new model below.")
...
predict_x = model.predict(test_sample) 
classes_x = np.argmax(predict_x, axis=1)
...
# display confusion matrix
confusion_matrix = pd.crosstab(test_label_compare, classes_x, rownames=['Actual'], colnames=['Predicted'], margins=True)
```

# ![Python](https://github.com/fahim19dipu/MI-Classifer/blob/main/images/icon.png) MI-Classifer


# Introduction
The Brain-Computer Interface (BCI) is a technology that uses brain impulses to allow direct communication with a machine. BCIs are most commonly employed in medical applications, such as neural control of prosthetic artificial limbs. Recent research, frequently using noninvasive techniques based on electroencephalography, has paved the way for new BCIs aimed at improving the performance of healthy users.
EEG analysis has been a significant tool in neuroscience, with applications in neuroscience, neural engineering, and even commercial applications. Many of the analytical techniques used in EEG investigations have incorporated machine learn- ing to identify useful information for brain categorization and neuroimaging. Deep learning architectures have been deployed as a result of the availability of large EEG data sets and recent breakthroughs in machine learning, particularly in the interpretation of EEG signals and comprehending the knowledge it will hold for brain functionality. The accurate automatic classification of these signals is an important step toward making EEG more practical in a wide range of applications and much less reliant on trained professionals.
The use of electroencephalography (EEG) data for motor imagery-based brain-computer interface (MI-BCI) has received a lot of attention in the past few decades, but the biggest challenge in BCI is obtaining reliable classification performance of the MI tasks.

## Project Aim
This project aims to develop a software framework that shows classification performance in various representation methods of different classifiers based on preprocessing methods and parameters based on user input for a set of training and testing datasets that contains EEG recordings of motor imagery tasks.


##Dataset

We chose to use Data Set 2a from BCI Competition IV[3] for our study. The number of control classes in Data Set 2a is increased from two to four, posing the danger of a drop in classification accuracy. However, it has the potential to provide greater information transfer speeds and more natural user-application interaction paradigms. In combination with the continuous categorization setting, this is clearly of practical importance.

Electroencephalographic (EEG) data from nine people are included in this data set. The BCI cue-based paradigm consists of four separate motor imagery tasks: left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4). (class 4). For each subject, two sessions were recorded on different days. Each session consists of six runs with short pauses in between. A single run contains 48 trials (12 for each of the four classes), for a total of 288 trials each session.

### 4.1 Proposed method

The total process of the work that has been done here can be summed up in
following way which is also shown graphically in 4.1.1.

- Preprocessing
    - Data extraction
       ∗ Class selection
       ∗ Channel selection
       ∗ Bandpass filter application
    - Feature extraction
    - Feature selection
- Classification
- Result representation

### 4.2 Preprocessing

In this step, the data is processed before the classifier is to be trained and tested. Brain signals are generally very noisy and they contain some additional information that hampers the classifier’s classification ability. Also, the maximum portion of desired information can be found in a certain band of frequency, So the signals must be processed before being feed into the classifier.
They are described below.


```
Figure 4.1.1: Working procedure
```
#### 4.2.1 Data Extraction

In this step, we have applied here are class selection, channel selection, and bandpass filter application to keep only the classes, channels, and frequency band we need rather than taking all the available data. This results in a less redundant dataset.

Class Selection

As the dataset, we are working with has a total of 4 classes. We have provided the users with the choice of selecting any combination of classes that they desire. If the user chooses manual task selection, they will be given this option and the classifiers will be trained and tested with only the samples that are of the classes that the user selected. Otherwise, the classifiers will be trained and tested using all 4 classes.

Channel Selection

In the working dataset, BCI competition IV dataset 2a the EEG data was recorded through 22 channels. there were also 3 EOG channels. Here we are only working with EEG data, so we are giving the user option between choosing all the channels or some of them.
If the user chooses manual channel selection only data from the selected channels will be considered for the training and testing of the machine learning model. Otherwise, the model will be trained with data from all 22 channels.


Bandpass Filter Application

Applying the bandpass filter is almost an essential part of classifying any kind of EEG data as certain information about certain tasks is mostly found in certain frequency bands. In our background study, we have found that mostly two bandpass filters have been used while working with dataset 2a of the BCI Competition IV, Butterworth, and Chebychev. We have given the user to choose any one of the two filters as
they deem appropriate.
In the case of selecting a frequency band, the user has full liberty to use any frequency band they want as long as the highcut does not exceed the sampling frequency. But they can only use one band. When passing the data through the bandpass filter, we have found that the frequency band 4 Hz to 35 Hz shows better accuracy in all the cases.

#### 4.2.2 Feature Extraction

In the EEG data collected from the scalp different types of frequency components get added to the electrical signal of the brain. But more importantly, we don’t necessarily need al the components. With feature selection, we try to get the frequency components that are more dominant in all the data and that have a better correlation to the output.

We have decided to use the Common spatial pattern(CSP) as the feature extraction method in this work. the user can decide how many features are to be extracted from the data. it can be as many as the number of selected channels to 1.

#### 4.2.3 Feature Selection

After feature extraction, the process of selecting the best features among the extracted feature set is called feature selection. It is used to reduce the dimensionality of the data and train the data with only the best features. Thus improving the accuracy and training time.

The feature selection method we have used is the SelectKBest method. This method chooses the features in the dataset that contributes most to the label or the target variable. The user can choose the number of features to keep and it must be less than the number of extracted features.


### 4.3 Classification

In this step after the data has been preprocessed and the features have been extracted, the feature set is to be trained and tested. But which classification algorithm is to be implemented is based on the user’s choice. Sequentially all the selected classification algorithm is implemented and the preprocessed data is used to train and test the classification model. 
The algorithms we are going to implement are

- Convolutional Neural Network (CNN)
- Artificial Neural Network (ANN)
- Deep Belief Network (DBN)
- K Nearest Neighbor (KNN)
- Linear Discriminant Analysis (LDA)
- Random Forest
- Support Vector Machine (SVM)

#### 4.3.1 Convolutional Neural Network (CNN)

A convolutional neural network (CNN) is a type of artificial neural network that uses deep learning to perform both generative and descriptive tasks, often used in computer vision that includes image and video recognition, along with recommender systems and natural language processing (NLP). It has also been used in EEG classification. The CNN model we have used consists of
- 2 convolutional layers, each with a max-pooling layer
- 1 dense layer
- 1 Output layer
Convolutional Layers: In the first convolutional layer, the number of output filters in the convolution is 128, kernel size 3x3, same padding and applies relu activation function. The max-pooling layer of this layer is of stride 2 and pool size of 2x2.
In the second convolutional layer, the number of output filters in the convolution is 64, kernel size 3x3, same padding and applies relu activation function. The max-pooling layer of this layer is the same as the previous of stride 2 and pool size of 2x2.

Hidden Layer: For the dense layer we used has 1000 units and uses relu activation function. It also uses l2 regularization for better generalization for real-world data.

Output Layer: For the output layer we have 5 units and it uses the softmax activation function.

For compilation, we used the adam optimizer as the optimizer, sparse categorical cross-entropy as the loss function.

#### 4.3.2 Artificial Neural Network (ANN)

Artificial neural network (ANN) is the piece of a computing system designed to simulate the way the human brain analyzes and processes information. It is the foundation of artificial intelligence (AI) and solves problems that would prove impossible or difficult by human or statistical standards. ANNs have self-learning capabilities that enable them to produce better results as more data becomes available.

The parameters with which the model has been trained are following

- Hidden layers structure = [1500, 500, 5100]
- Learning rate = 0.001
- Loss function = sparse categorical crossentropy
- Epochs = 25
- Batch size = 32
- Activation function = relu
- validation split=0 .2

#### 4.3.3 Deep Belief Network (DBN)

Deep belief network (DBN) is a generative graphical model, or alternatively a class of deep neural networks, composed of multiple layers of latent variables (”hidden units”), with connections between the layers but not between units within each layer.

When trained on a set of examples without supervision, a DBN can learn to probabilistically reconstruct its inputs. The layers then act as feature detectors. After this learning step, a DBN can be further trained with supervision to perform classification.


The parameters with which the model has been trained are following

- Hidden layers structure =[500, 500,500]
- Learning rate of restrcited bolzman mechine (RBM) =0.01
- Learning rate=0.01
- Number of epochs of rbm=10
- Number of iteration (backprop)=50
- Batch size=32
- Sctivation function= relu
- Dropout rate=0.2

#### 4.3.4 Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis also known as Normal Discriminant Analysis or Discriminant Function Analysis is a dimensionality reduction technique that is commonly used for supervised classification problems. It is used for modeling differences in groups i.e. separating two or more classes. It is used to project the features in higher dimension space into a lower dimension space.

#### 4.3.5 Random Forests

Random Forest is a flexible, easy-to-use machine learning algorithm that produces, even without hyper-parameter tuning, a great result most of the time. It is also one of the most used algorithms because of its simplicity and the fact that it can be used for both classification and regression tasks. It creates a forest and makes it somehow random. The ”forest” it builds, is an ensemble of Decision Trees, most of the time trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.

One big advantage of random forest is, that it can be used for both classification and regression problems, which form the majority of current machine learning systems. We have trained the system with 1000 estimators and 0 random states.


#### 4.3.6 Support-Vector Machine (SVM)

In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting).

An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.

Various variations of SVM are available. These variations are created by using different Kernels which are a set of mathematical functions to take data as input and transform it into the required form. Here we have used three of those variations.

- SVM using Linear kernel
- SVM using Polynomial kernel
- SVM using Gaussian radial basis function(RBF) kernel

#### 4.3.7 K Nearest Neighbour (KNN)

K-nearest neighbors algorithm (k-NN) is a non-parametric classification method first developed by Evelyn Fix and Joseph Hodges in 1951 and later expanded by Thomas Cover. It is used for classification and regression. In both cases, the input consists of the k closest training examples in the data set. The output depends on whether k-NN is used for classification or regression.

In the k-NN classification which we are applying, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. For our model, we have selected 7 as the value of K.


### 4.4 Result Representation

The result here refers to the performances of the classification algorithms as well as used preprocessing methods. This performance is usually measured in various matrices. The metrics we have used are accuracy and kappa score.

#### 4.4.1 Accuracy

Classification Accuracy is what we usually mean when we use the term accuracy. It is the ratio of the number of correct predictions to the total number of input samples. It works well only if there are an equal number of samples belonging to each class.

```
Accuracy=Number of total predictions made / Number of correct predictions (4.1)
```
#### 4.4.2 Kappa Value

Kappa value is a statistical coefficient that is used to measure inter-rater reliability for categorical items[39]. It is generally thought to be a more robust measure than simple percent agreement calculation, as it takes into account the possibility of the agreement occurring by chance.

```
k= 1− 11 −−the hypothetical probability of chance agreement relative observed agreement among raters (4.2)
```
#### 4.4.3 Confusion Matrix

Confusion Matrix is a performance measurement for machine learning classification. A confusion matrix, also known as an error matrix[40], is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one (in unsupervised learning it is usually called a matching matrix). Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class[41].


Figure 4.4.1: Confusion matrix for binary classification



# IMPLEMENTATION



The implementation phase of the project is the development of the designs produced during the design phase Through a series of screenshots, code snippets, and descriptions this section will show how we are trying to meet the generic requirements of the application, and where necessary the differences between the proposed designs and actual implementation.

### 5.2 Interface

 This design was chosen and implemented because of its clear layout and navigation structure. The final interface can be viewed in 5.2.1.


```
Figure 5.2.1: The screenshot of the final UI
```
### 5.3 Navigation

The main navigation is done through buttons. The UI is divided into several sections. Each section contains its distinctive buttons, radio buttons, and checkboxes. The navigation and explanation of each section are described below.

#### 5.3.1 Dataset Spacification

This section asks for necessary instruction from the user about the dataset which is to be selected. Tasklist collection indicates the task information be collected manually or automatically. Initially, it is selected as ”Auto”. If ”Manual” is selected then the class checkboxes will appear and collect which of the classes of the dataset is to be selected. By clicking the ”Select File” button the user can select a dataset file from the computer and the path of the file is shown in the field Dataset path. There is also a combo box that gives the user opportunity to select the file type of the dataset.


#### 5.3.2 Channel Selection

This section asks for necessary instruction from the user about the channels which are present in the dataset. Selecting the button ”All Channels” results in the selection of all 22 channels. Otherwise through ”Manual Selection” the user can select necessary channels. By clicking the ”Manual selection” a pop-up window will open which is shown in 5.3.1.

```
Figure 5.3.1: Manual Selection Pop up window
```
Here we can see all available channels can be selected by checking their respective checkbox which will also result in the lighting up of the channel to green in the image on the right side of the window.it is shown in figure 5.3.2

```
Figure 5.3.2: Selecting channels in Pop up window
```
To reselect all the channels the user can click the button ”Check All” which is shown in figure 5.3.3.


```
Figure 5.3.3: checking all in Manual Selection
```
To unselect all the channels the user can click the button ”Uncheck All” which is shown in figure 5.3.4. After finishing the channel selection, clicking the okay button will save the selected channel list and close the pop-up window.

```
Figure 5.3.4: Unchecking all in Manual Selection
```
#### 5.3.3 Bandpass Filter

This section asks for necessary instruction from the user about the bandpass filter specification. Selecting ”Butterworth” will result in the data be bandpass filtered using the” Butterworth” bandpass filter. Similarly selecting ”Chebyshev II” will result in the data be bandpass filtered using forward-reverse ”Chebyshev” bandpass filter. The lowcut and highcut of the frequency range are taken from the entry box. The order and sampling frequency are also taken from here.


#### 5.3.4 Feature Engineering

This section asks for necessary instruction from the user about the feature extraction and selection. The” Feature Extraction (CSP)” part asks for the ”No. of Components” which is the number of features to be extracted from the EEG signal using CSP. The” Feature Selection (KBest)” part asks for the ”No. of Features” which is the number of Features to be kept from the available feature set.

#### 5.3.5 Classification Metrics

This section asks whether the classification report be shown as ”Accuracy” or ”Kappa Score”. It is to be selected as radio buttons.

#### 5.3.6 Classification Report

This section asks the user to select the models on which the dataset will be trained and tested. The user can select multiple models. The interface after all options has been selected can be viewed in 5.3.5.

```
Figure 5.3.5: UI after Selection
```

#### 5.3.7 Execution

After selecting All the necessary options and clicking the execute button the program will run and a progress bar will appear as shown in figure 5.3.6.

```
Figure 5.3.6: Progress bar
```
After finishing execution a message box will appear which will show the sample, class, and channels information shown in figure 5.3.7.

```
Figure 5.3.7: Trained dataset info
```
After finishing execution we can also see that the ”Classification Report” section has changed which is shown in figure 5.3.8.

```
Figure 5.3.8: Trained dataset info
```
If we select any ”Confusion Matrix” button, the confusion matrix of the respective classifier will appear which is shown in figure 5.3.9.


Figure 5.3.9: Confusion Matrix






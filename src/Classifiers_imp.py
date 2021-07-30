# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 22:00:12 2021

@author: fahim
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:55:08 2021

@author: fahim
"""

#import os

import numpy as np
from numpy import newaxis
from datetime import datetime 
from os import system


from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.metrics import classification_report,  accuracy_score ,confusion_matrix,cohen_kappa_score
from sklearn.feature_selection import SelectKBest,mutual_info_classif
#from sklearn.feature_selection import chi2,
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from scipy.signal import butter, lfilter
from scipy.io import loadmat
from scipy import signal

import tensorflow as tf
from tensorflow.keras import regularizers
#from tensorflow.keras.utils import np_utils

from mne.decoding import CSP
from dbn.models import SupervisedDBNClassification

import matplotlib.pyplot as plt
import itertools

import xgboost as xgb 
import lightgbm as lgb
##############################################################
def clear():
    _ = system('cls')
# def clear():
#     # check and make call for specific operating system
#     _ = call('cls')

""" Plot confussion matrix and save as image"""
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None, normalize=None):
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #return plt
    plt.tight_layout()
    
    plt.savefig('images/cf.png',pad_inches=0.1)

"""Implementing  Chebyshev type-2"""
def cheby_imp(sig,l,h,fs):

    """Specification""" 
    Fs = fs                                     # Sampling frequency in Hz 
    fp = np.array([l, h])                       # Pass band frequency in Hz 
    fs = np.array([l-1, h+1])                   # Stop band frequency in Hz
    Ap = 1                                      # Pass band ripple in dB 
    As = 2                                      # Stop band attenuation in dB 
    wp = fp/(Fs/2)                              # Normalized passband edge frequencies w.r.t. Nyquist rate  
    ws = fs/(Fs/2)                              # Normalized stopband edge frequencies 
    
    """Compute order of the Chebyshev type-2"""  
    N, wc = signal.cheb2ord(wp, ws, Ap, As)      # digital filter using signal.cheb2ord 
    #'Order of the filter=', N                     Cut-off frequency=', wc

    """Design digital Chebyshev type-2 bandpass"""  
    # filter using signal.cheby2 function 
    sos= signal.cheby2(N, As, wc, 'bandpass',Fs, output='sos') 
    #sos = signal.cheby2(12, 20, 17, 'hp', )
    filtered = signal.sosfiltfilt(sos, sig)
    return filtered

""" applying butterworth banpass filter"""
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    #b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

""" Extract data from matlab files and applies selected bandpass filter"""
def extract_Data(filename,lowcut, highcut, fs, order,cheby_c,classes,ch_list):
    
    X=np.zeros(0).reshape(0,ch_list.shape[0],750)                              # creating the dataset variable
    Y=np.zeros(0) 

    A01E = loadmat(filename,struct_as_record=True, squeeze_me=False)            # loading matfile
    data = A01E['data']         

    j=0
    while(j<6):
        if(filename == "F:/4th year project resource/matlab dataset/A04T.mat"):   # exception for A04T
            runn= data[0,j+1]
        else:
            runn= data[0,j+3]                                                     #selecting the run
            
        trial = runn["trial"][0][0][:,0]                         # reconfiguring trail values to python Compatable
        y = runn["y"][0][0][:,0]                                 # reconfiguring the LABEL to python Compatable

        for val in classes:                                      # Excluding classes
            y_exc_loc=np.where(y==val)      
            y=np.delete(y,y_exc_loc[0],0)                       # Excluding labels
            trial=np.delete(trial,y_exc_loc[0],0)               # Excluding trails
                
        i=0
        while(i<len(trial)):                             # for all remeaining trails extract the respective sample values 
        
            x = runn["X"][0][0][:,ch_list]     # reconfiguring the SIGNAL VALUE of the selected channels
            x=x[trial[i]+749:trial[i]+1499:1]  # Taking the values from the traul position to tvhe next 750 values
            
            """Reshaping the data"""
            x= x.transpose()
            x = x[newaxis,:]  
            
            "applying bandpass"
            if(cheby_c==0):
                x = butter_bandpass_filter(x, lowcut, highcut, fs, order)
            else:
                x = cheby_imp(x, lowcut, highcut, fs)
            
            """Concarting the values"""    
            X=np.append(X,x,axis=0)   #Axis 0 stack on top of another
            i+=1
        
        Y=np.append(Y,y,axis=0)
        j+=1

    Y=Y.astype(int)
    return X,Y

""" Extract data , apply badpass filter, Apply Csp(Feature Extraction), Apply Feature selection(Kbest) 
    Returns Feature set of both training and testing data"""
def preprocess_data(train_filename,test_filename,cheby_c,lowcut,highcut,order,fs,
                    n_of_comp,n_of_ft,classes,ch_list):

    #########################################################   Extract data
    """ Find out which of the classes to be excluded"""
    classes=np.where(classes==0)
    classes=classes[0]
    classes +=1                                                             # Classes to exclude
    
    # print("The incoming data are")
    # print(train_filename,test_filename,cheby_c,lowcut,highcut,order,fs,
    #                 n_of_comp,n_of_ft,classes,ch_list)

    """Extract sampled signal values for all the trails of selected classes and apply bandpassfilter""" 
    
    X_ORG,Y=extract_Data(train_filename,lowcut,highcut,fs,order,cheby_c,classes,ch_list)

    #print("test processs")
    X_test_ORG,Y_test=extract_Data(test_filename,lowcut,highcut,fs,order,cheby_c,classes,ch_list)
    
    ########################################################   CSP wth Average power
    """Appltying CSP to train and test data"""
    
    csp_alt= CSP(n_of_comp,reg=None, log=None, norm_trace=False)

    #print("csp")
    ft_alt=csp_alt.fit_transform(X_ORG,Y)
    ft_T_alt=csp_alt.fit_transform(X_test_ORG,Y_test)
    
    """Feature selection"""
     
    ft_alt= SelectKBest(mutual_info_classif, k=n_of_ft).fit_transform(ft_alt,Y)
    ft_T_alt= SelectKBest(mutual_info_classif, k=n_of_ft).fit_transform(ft_T_alt,Y_test)
    
    return ft_alt,Y,ft_T_alt,Y_test,len(classes),X_ORG.shape[0],X_ORG.shape[1]


#%%                        
"""                         All classifers return either accuracy or kappa value along with Confusion matrix    
"""
"""                                                   Classifier              Light Gbm             """
def lightgbm(ft_alt,Y,ft_T_alt,Y_test,is_keppa):
        
    # Encoding categorical data
    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(Y) # M=1 and B=0
    
    #Define x and normalize values
    scaler = StandardScaler()
    scaler.fit(ft_alt)
    X_train = scaler.transform(ft_alt)
    
    #####################################################
    #Light GBM
        
    d_train = lgb.Dataset(X_train, label=y_train)
    
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    #Try dart for better accuracy
                  'objective':'multiclass',
                  'metric':[ 'multi_logloss'],
                  'num_leaves':100,
                  'max_depth':10,
                  'num_class':4}
    start=datetime.now()
    
    clf = lgb.train(lgbm_params, d_train, 50) #50 iterations. Increase iterations for small learning rates
    
    stop=datetime.now()
    
    execution_time_lgbm = stop-start
    print("LGBM execution time is: ", execution_time_lgbm)
    
    #Prediction on test data
    scaler = StandardScaler()
    scaler.fit(ft_T_alt)
    X_test = scaler.transform(ft_T_alt)
    
    labelencoder = LabelEncoder()
    y_test = labelencoder.fit_transform(Y_test)
    
    
    y_pred_lgbm=clf.predict(X_test)
    y_pred_lgbm = np.array([np.argmax(i) for i in y_pred_lgbm])
           
    #Print accuracy
    acc= metrics.accuracy_score(y_pred_lgbm,y_test)
    print ("Accuracy with LGBM = ",acc)
           
    #Confusion matrix
    
    cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
    
    kappa = cohen_kappa_score( y_test, y_pred_lgbm)
    print("kAPPA score = ",kappa )
    if(is_keppa==0):
        return round(acc*100,3),cm_lgbm
    else:
        return round(kappa,3),cm_lgbm
"""                                                   Classifier              XGBoost            """

def xgb_i(ft_alt,Y,ft_T_alt,Y_test,is_keppa):
        
    # Encoding categorical data
    labelencoder = LabelEncoder()
    y_train = labelencoder.fit_transform(Y) # M=1 and B=0
    
    labelencoder = LabelEncoder()
    y_test = labelencoder.fit_transform(Y_test)    

    #Define x and normalize values
    scaler = StandardScaler()
    scaler.fit(ft_alt)
    X_train = scaler.transform(ft_alt)
    
    scaler = StandardScaler()
    scaler.fit(ft_T_alt)
    X_test = scaler.transform(ft_T_alt)
    
    #####################################################
    #Extreme GRadient Boosting
        
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test)
    
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    xgb_params = {'max_depth':10, 
                  'objective':'multi:softmax',
                  'metric':'mlogloss',
                  'learning_rate':.05,
                   'num_class':4}
    
    start=datetime.now()
    
    clf = xgb.train(xgb_params, d_train, 50) #50 iterations. Increase iterations for small learning rates
    
    stop=datetime.now()
    
    execution_time_xgb = stop-start
    print("XGBoost execution time is: ", execution_time_xgb)
    
    #Prediction on test data
    y_pred_xgb=clf.predict(d_test)
    #y_pred_xgb = np.array([np.argmax(i) for i in y_pred_xgb])
           
    #Print accuracy
    acc= metrics.accuracy_score(y_pred_xgb,y_test)
    print ("Accuracy with XGB = ",acc)
           
    #Confusion matrix
    
    cm_xgb= confusion_matrix(y_test, y_pred_xgb)
    
    kappa = cohen_kappa_score( y_test, y_pred_xgb)
    print("kAPPA score = ",kappa )
    if(is_keppa==0):
        return round(acc*100,3),cm_xgb
    else:
        return round(kappa,3),cm_xgb

"""                                                   Classifier  -    SVM(kernel = Linear)           """
def cls_svm_lin(ft_alt,Y,ft_T_alt,Y_test,is_keppa):

    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(ft_alt, Y)
    accuracy_lin = linear.score(ft_T_alt, Y_test)
    print("\n\nAccuracy Linear Kernel:", accuracy_lin)                        # mean accuracy results
    linear_pred = linear.predict(ft_T_alt)
    
    cm_lin = confusion_matrix(Y_test, linear_pred)                            # confusion matrix  
    print(cm_lin)
    
    keppa_svm_lin=cohen_kappa_score(Y_test, linear_pred)                      # Cohen Kappa score
    print(keppa_svm_lin)
    
    
    if(is_keppa==0):
        return round(accuracy_lin*100,3),cm_lin
    else:
        return round(keppa_svm_lin,3),cm_lin

"""                                                   Classifier  -    SVM(kernel = Plinomial)      """
def cls_svm_poli(ft_alt,Y,ft_T_alt,Y_test,is_keppa):
    
    poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(ft_alt, Y)
    accuracy_poly = poly.score(ft_T_alt, Y_test)
    print("\n\nAccuracy Polinomial Kernel:", accuracy_poly)                   # mean accuracy results
    poly_pred = poly.predict(ft_T_alt)
    
    cm_poly = confusion_matrix(Y_test, poly_pred)                              # confusion matrix  
    print(cm_poly)
    
    keppa_svm_poly=cohen_kappa_score(Y_test, poly_pred)                       # Cohen Kappa score
    print(keppa_svm_poly)
    
    if(is_keppa==0):
        return round(accuracy_poly*100,3),cm_poly
    else:
        return round(keppa_svm_poly,3),cm_poly

"""                                                   Classifier  -    SVM(kernel =  RBF)         """
def cls_svm_rbf(ft_alt,Y,ft_T_alt,Y_test,is_keppa):
    
    rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(ft_alt, Y)
    accuracy_rbf = rbf.score(ft_T_alt, Y_test)
    print("\n\nAccuracy RBF Kernel:", accuracy_rbf)                            # mean accuracy results
    rbf_pred = rbf.predict(ft_T_alt)
    
    
    cm_rbf = confusion_matrix(Y_test, rbf_pred)                                 # confusion matrix  
    print(cm_rbf)
    
    keppa_svm_rbf=cohen_kappa_score(Y_test, rbf_pred)                          # Cohen Kappa score
    print(keppa_svm_rbf)
    if(is_keppa==0):
        return round(accuracy_rbf*100,3),cm_rbf
    else:
        return round(keppa_svm_rbf,3),cm_rbf
"""                                                   Classifier  -           LDA            """
def cls_lda(ft_alt,Y,ft_T_alt,Y_test,is_keppa):

    lda = LinearDiscriminantAnalysis()
    lda.fit(ft_alt, Y)
    accuracy_LDA = lda.score(ft_T_alt,Y_test)
    print("\n\nAccuracy LDA",accuracy_LDA)                                     # mean accuracy results
    LDA_pred=lda.predict(ft_T_alt)
    
    cm_lda=confusion_matrix(Y_test, LDA_pred)                                  # confusion matrix  
    print(cm_lda)
    
    keppa_LDA=cohen_kappa_score(Y_test, LDA_pred)                              # Cohen Kappa score 
    print(keppa_LDA)
    
    if(is_keppa==0):
        return round(accuracy_LDA*100,3),cm_lda
    else:
        return round(keppa_LDA,3),cm_lda

"""                                                   Classifier  -           KNN                """ 
def cls_knn(ft_alt,Y,ft_T_alt,Y_test,is_keppa):
    
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(ft_alt,Y)
    y_pred = model.predict(ft_T_alt)

    CM_knn= confusion_matrix(Y_test,y_pred)                                     # confusion matrix  
    print(CM_knn)
    
    CR_knn=classification_report(Y_test,y_pred)                                # Classification report
    print(CR_knn)
    
    acc_knn=accuracy_score(Y_test, y_pred)
    print("\n\nAccuracy KNN:",acc_knn)                                          # mean accuracy results
    
    keppa_knn=cohen_kappa_score(Y_test, y_pred)                                # Cohen Kappa score
    print(keppa_knn)
    
    if(is_keppa==0):
        return round(acc_knn*100,3),CM_knn
    else:
        return round(keppa_knn,3),CM_knn
    
"""                                                   Classifier  -    Random forset          """
def cls_rf(ft_alt,Y,ft_T_alt,Y_test,is_keppa):
    
    sc = StandardScaler()
    X_train = sc.fit_transform(ft_alt)
    X_test = sc.transform(ft_T_alt)
    print("ok")   
    regressor = RandomForestClassifier()
    regressor.fit(X_train, Y)
    y_pred = regressor.predict(X_test)
    
    CM_rf= confusion_matrix(Y_test,y_pred)                                     # confusion matrix  
    print(CM_rf)
    
    CR_rf=classification_report(Y_test,y_pred)                                 # Classification report 
    print(CR_rf)
    
    acc_rf=accuracy_score(Y_test, y_pred)
    print("\n\nAccuracy Randomm forest:",acc_rf)                               # mean accuracy results
    
    keppa_rf=cohen_kappa_score(Y_test, y_pred)                                 # Cohen Kappa score
    print(keppa_rf)
    if(is_keppa==0):
        return round(acc_rf*100,3),CM_rf
    else:
        return round(keppa_rf,3),CM_rf
"""                                                   Classifier  -           DBN           """
def cls_dbn(ft_alt,Y,ft_T_alt,Y_test,is_keppa):

    ss=StandardScaler()
    X = ss.fit_transform(ft_alt)
    X_t = ss.fit_transform(ft_T_alt)

    classifier = SupervisedDBNClassification(hidden_layers_structure =[500, 500,500], 
                                             learning_rate_rbm=0.01, 
                                             learning_rate=0.01, 
                                             n_epochs_rbm=10, 
                                             n_iter_backprop=50, batch_size=32, 
                                             activation_function='relu', 
                                             dropout_p=0.2)
    classifier.fit(X, Y)
    y_pred = classifier.predict(X_t)
    acc_dbn=accuracy_score(Y_test, y_pred)
    print('\nAccuracy of Prediction: ',acc_dbn )                               # mean accuracy results
    
    cm_dbn=confusion_matrix(Y_test, y_pred)                                    # Confusion Matrix
    
    keppa_dbn=cohen_kappa_score(Y_test, y_pred)                                # Cohen Kappa score
    print(keppa_dbn)
    
    if(is_keppa==0):
        return round(acc_dbn*100,3),cm_dbn
    else:
        return round(keppa_dbn,3),cm_dbn

"""                                                   Classifier  -           ANN             """
def cls_ann(ft_alt,Y,ft_T_alt,Y_test,is_keppa):
    """ Lanesl encoding"""
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    Y_test = le.transform(Y_test)
    """Model parameters"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),       
        
        tf.keras.layers.Dense(1500, activation=tf.nn.relu),
        tf.keras.layers.Dense(500, activation=tf.nn.relu),
        tf.keras.layers.Dense(5100, activation=tf.nn.relu),
        tf.keras.layers.Dense(1500, activation=tf.nn.relu),
        # tf.keras.layers.Dense(1500, activation=tf.nn.relu),
        # tf.keras.layers.Dense(1500, activation=tf.nn.relu),
        
        # tf.keras.layers.Dense(1500, activation=tf.nn.relu),
        # tf.keras.layers.Dense(500, activation=tf.nn.relu),
        # tf.keras.layers.Dense(5100, activation=tf.nn.relu),
        # tf.keras.layers.Dense(1500, activation=tf.nn.relu),
        # tf.keras.layers.Dense(1500, activation=tf.nn.relu),
        # tf.keras.layers.Dense(1500, activation=tf.nn.relu),
                   
        tf.keras.layers.Dense(len(np.unique(Y))+1,  activation=tf.nn.softmax)
    ])
    
    model.compile(tf.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(ft_alt, Y, epochs=25,batch_size=32,validation_split=.2,shuffle=True)
    test_predictions = model.predict(ft_T_alt)
    test_loss,nn_accuracy =model.evaluate(ft_T_alt,Y_test,batch_size=32)
    print("\n\nAccuracy on ANN : ",nn_accuracy)                             # mean accuracy results
          
    print("Classification report for classifier\n")                        # Classofocation report
    print(metrics.classification_report(Y_test, np.argmax(test_predictions,axis=1) ))
    
    confusion_ann = confusion_matrix(Y_test, np.argmax(test_predictions,axis=1))
    print(confusion_ann)                                                  # Confusion Matrix
    
    keppa_ann=cohen_kappa_score(Y_test, np.argmax(test_predictions,axis=1))   
    print(keppa_ann)                                                      # Cohen Kappa score
    
    if(is_keppa==0):
        return round(nn_accuracy*100,3),confusion_ann
    else:
        return round(keppa_ann,3),confusion_ann
    
"""                                                   Classifier              CNN             """
def cls_cnn(train_filename,test_filename,cheby_c,lowcut,highcut,order,fs,n_of_comp,n_of_ft
           ,is_keppa,classes,ch_list):
    """ Find out which of the classes to be excluded"""
    classes=np.where(classes==0)
    classes=classes[0]
    classes +=1 
    
    print(classes)
    """As CNN reqires 4 dimentional data  Preprocessing is done separately"""
    X_ORG,Y=extract_Data(train_filename,lowcut,highcut,fs,order,cheby_c,classes,ch_list)
    X_test_ORG,Y_test=extract_Data(test_filename,lowcut,highcut,fs,order,cheby_c,classes,ch_list)
    
    ##############################################   Apply CSP in CSP space {used in CNN}
    """Apply CSP in CSP space {only used in CNN}  """
    
    csp = CSP(n_of_comp, reg=None, log=None, norm_trace=False, transform_into='csp_space')
    
    ft=csp.fit_transform(X_ORG,Y)
    ft_T=csp.fit_transform(X_test_ORG,Y_test)
    
    ft = ft[...,newaxis] 
    ft_T = ft_T[...,newaxis]
 
    """Labels encoding"""   
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    Y_test = le.transform(Y_test) 
    """ Model Parameter"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
     
        #,kernel_regularizer=regularizers.l2(0.0001)
        tf.keras.layers.Flatten(),    
    #    tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.0001)),
        tf.keras.layers.Dense(len(np.unique(Y))+1,  activation=tf.nn.softmax)
    ])
    
    model.compile(tf.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(ft, Y, epochs=25,batch_size=32,validation_split=.2,shuffle=True)
    test_predictions = model.predict(ft_T)
    test_loss,cnn_accuracy =model.evaluate(ft_T,Y_test,batch_size=32)
    print("\n\nAccuracy on CNN : ",cnn_accuracy)                             # mean accuracy results

    keppa_cnn=cohen_kappa_score(Y_test, np.argmax(test_predictions,axis=1))  
    print(keppa_cnn)                                                         # mean COhen Kappa score
    
    print("Classification report for classifier\n")                           # Classification Report
    print(metrics.classification_report(Y_test, np.argmax(test_predictions,axis=1) ))
    
    confusion_cnn = confusion_matrix(Y_test, np.argmax(test_predictions,axis=1))
    print(confusion_cnn)                                                     # Confusion matrix
    
    if(is_keppa==0):
        return round(cnn_accuracy*100,3),confusion_cnn
    else:
        return round(keppa_cnn,3),confusion_cnn

#%%
"""                                                       Driver portion """

# file='F:/projects/4 MI Classifier/selected_Channels.txt' 
# #"""Each line of theb  file represeents the repective channel-index and Contains 0/1 for unselected or selected"""        
# with open(file) as f:
#     content = f.readlines()

# content = [x.strip() for x in content]

# channels = np.array(content)                                                    
# #channels = channels.astype(int)

# channels=np.where(channels=='1')                                            #list of channels (Manual)
# channels=channels[0]


# ALL_channels=np.arange(0,22,1)                                                     #list of channels (All )


# classes=np.array([1,1,1,1])                                                    #Selected classes
# is_keppa=0                                                              # 0 for accuracy /1 for keppa
# cheby_c=0                                                             # 0 for butterworth /1 for chebyshev
# ch_all =0


# # bandpass specification
# lowcut= 4
# highcut= 35
# fs= 250
# order=4

# n_of_comp=6                                                             # number of components for CSP
# n_of_ft=4                                                               # number of features for Kbest

# train_filename="F:/4th year project resource/matlab dataset/A09T.mat"
# test_filename="F:/4th year project resource/matlab dataset/A09E.mat"
# # # # #

# if ch_all == 1 :
#     channels = ALL_channels

# print(channels.shape)

# ft_alt,Y,ft_T_alt,Y_test,class_no,size,channel = preprocess_data(train_filename,test_filename,cheby_c,
#                                             lowcut,highcut,order,fs,n_of_comp,n_of_ft,classes,channels)

# """names of the selected classes"""
# class_names=np.where(classes==1)
# class_names=class_names[0]
# class_names +=1 


# x_xgb,cf=xgb_i(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_xgb)


# x_lgbm,cf=lightgbm(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_lgbm)


# x_lin,cf=cls_svm_lin(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_lin)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)


# x_knn,cf=cls_knn(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_knn)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)


# x_rf,cf=cls_rf(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_rf)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)


# x_lda,cf=cls_lda(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_lda)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)


# # x,cf=cls_svm_lin(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# # print("Res=",x)
# # print("Confusion matrix=\n",cf)
# # plot_confusion_matrix(cf,class_names)


# x_poli,cf=cls_svm_poli(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_poli)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)


# x_rbf,cf=cls_svm_rbf(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_rbf)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)


# x_ann,cf=cls_ann(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_ann)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)


# x_dbn,cf=cls_dbn(ft_alt,Y,ft_T_alt,Y_test, is_keppa)
# print("Res=",x_dbn)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf,class_names)



# x_cnn,cf=cls_cnn(train_filename,test_filename,cheby_c,lowcut,highcut,order,fs,n_of_comp,n_of_ft
#             ,is_keppa,classes,channels)
# print("Res=",x_cnn)
# print("Confusion matrix=\n",cf)
# #plot_confusion_matrix(cf, class_names)

# clear()

# print(test_filename)

# print(channels.shape[0])

# print("4 - ", highcut)

# if cheby_c==0:
#     print(order)
# else:
#     print("cheby")

# print(x_cnn)
# print(x_ann)
# print(x_dbn)

# print(x_lda)
# print(x_rf)
# print(x_lin)
# print(x_poli)
# print(x_rbf)
# print(x_knn)
#print("Res=",x_cnn)

# print(x_xgb)
# print(x_lgbm)
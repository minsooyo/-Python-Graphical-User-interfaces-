

import tkinter as tk
from tkinter import *
from tkinter import ttk
from  sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, neighbors, metrics, mixture, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np


 

window = tk.Tk()

window.title('LOVE YOU Niraj')

# width x height + x_offset + y_offset:

window.geometry("600x400+100+100")

left_frame = tk.Frame(master=window, width = 300, height = 500, bg ="skyblue")
left_frame.grid(row=1, column=0, sticky ="w")

 
right_frame = tk.Frame(master=window, width = 300, height = 500, bg ="grey")
right_frame.grid(row=1, column=1, sticky ="w")

#Set font

myfont = "Arial, 10"

 

#Add a label
lbl_header = tk.Label(text="Clasification of Datasets", font= myfont, height=1)
lbl_header.place(x=80, y=15)

 
#Add a label
lbl = tk.Label(text="Choose a dataset",
               fg="navy", anchor="w", width=12, height=1, font= myfont)
lbl.place(x=0, y=40)


##########Radiobutton for dataset selection##########

var1 = tk.StringVar()

rb1 = tk.Radiobutton(window,text="iris", variable=var1,
                     value='iris', font=myfont)
rb1.place(x=100, y=40)
rb1.deselect()

rb2 = tk.Radiobutton(window,text="breast_cancer",
                     variable=var1, value='breast_cancer', font=myfont)
rb2.place(x=145, y=40)
rb2.deselect()

 
rb3 = tk.Radiobutton(window,text="wine",
                     variable=var1, value='wine', font=myfont)
rb3.place(x=245, y=40)

##########Radiobutton for Classification dataset##########

var2 = tk.StringVar()
lbl2 = tk.Label(text="Choose a classification",
               fg="navy", anchor="w", width=12, height=1, font= myfont)
lbl2.place(x=0, y=80)

rb1_1 = tk.Radiobutton(window,text="SVM", variable=var2,
                     value='SVM', font=myfont)
rb1_1.place(x=120, y=80)
rb1_1.deselect()

rb2_1 = tk.Radiobutton(window,text="KNN",
                     variable=var2, value='KNN', font=myfont)
rb2_1.place(x=200, y=80)
rb2_1.deselect()


##########Radiobutton for k_fold##########

k_fold = [3,5,7]

lbl3 = tk.Label(text="K value:", fg="navy", anchor="w",width=8,
                height=1, font=myfont)
lbl3.place(x=0,y=120)
combo_clusters = ttk.Combobox(values=k_fold, width=5)
combo_clusters.current(1)
combo_clusters.place(x=100,y=120)


#########  Load data and difine classifier for further usage #####

def run2(data, classifier):
    data_type = var1.get()
    classifier_type = var2.get()
    if  data_type == "iris":
        datasett = datasets.load_iris()
    elif data_type == 'wine':
        datasett = datasets.load_wine()
    elif data_type == 'breast_cancer':
        datasett = datasets.load_breast_cancer()
    X = datasett.data
    y = datasett.target
    class_names = datasett.target_names

 

   # Split the dataset into Training set (80%) and Testing set (20%). 

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.2, random_state=0)

    if classifier_type == "SVM":
        
        # cross validation is used to set the parameters
        
        parameters = [{'gamma': [0.00001,0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}]

        print()

        classifier = svm.SVC()
    
    elif classifier_type == "KNN":
       
        parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}]
       
        classifier = neighbors.KNeighborsClassifier()

    #here, GSCV is grid search cross validation
        
    gscv_classifier = GridSearchCV(
        estimator = classifier,
        param_grid = parameters,
        cv = 5,  # 5-fold cross validation
        scoring ='accuracy'
    )

    gscv_classifier.fit(X_train, y_train)

    print("Grid scores on validation set:")
    print() 
    
    mean = gscv_classifier.cv_results_['mean_test_score']
    
    st_dev = gscv_classifier.cv_results_['std_test_score']
    
    result = gscv_classifier.cv_results_['params']
    
    for mean, std, param in zip(mean, st_dev, result):
        print("Parameters: %r, accuracy: %0.2f (+/-%0.02f)" % (param, mean, std * 2))
    print()

    print("The best parameters:", gscv_classifier.best_params_)

    
    #### plot confusion matrix and accuracy vs parameter
    
    y_pred = gscv_classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred) * 100


    number_k = np.arange(1, 10)

    train_accuracy = np.empty(len(number_k))

    test_accuracy = np.empty(len(number_k))
  

    for i, k in enumerate(number_k):

        knn = neighbors.KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train,y_train)

        train_accuracy[i] = knn.score(X_train, y_train)

        test_accuracy[i] = knn.score(X_test, y_test)

    plt.subplot(1, 1, 1)
    plt.title('dataset classification accuracy')
    plt.plot(number_k, test_accuracy, label = 'Testing Accuracy')
    plt.plot(number_k, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Order of parameter')
    plt.ylabel('Accuracy')
    plotcm = metrics.plot_confusion_matrix(gscv_classifier, X_test, y_test, display_labels=class_names) 
    plotcm.ax_.set_title('Accuracy = {0:.2f}%'.format(accuracy))
    plt.show()


   
#########  Apply classifier and data together #####

def data():
    data = var1.get()
    classifier = var2.get()
    if data == 'iris':
        run2("iris", classifier)
    elif data == 'wine':
        run2("wine", classifier)
    elif data == 'breast_cancer':
        run2("breast_cancer",classifier)
      

    

#########  Add run button ############
button = tk.Button(window, text="RUN", fg="black",width=10,command=data)
button.place(x=10,y=200)
   
window.mainloop()
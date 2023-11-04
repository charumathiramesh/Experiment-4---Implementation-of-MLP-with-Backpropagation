# Experiment-4---Implementation-of-MLP-with-Backpropagation

## AIM:
To implement a Multilayer Perceptron for Multi classification

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes

  	(i)Feed Forward pass
	         (ii)Backward pass
           
Ø  Learning process –backpropagation

Ø  Computationally efficient method

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other
   
Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)



## ALGORITHM:

1.Import the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

5.In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6.Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7.In order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

## PROGRAM :
```
Developed by : CHARUMATHI R
ref no : 212222240021
```
```c
# Include necessary libraries
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('IRIS.csv')
df.head

names = ['sepal-length','sepal-width','petal-length','petal-width','Class']

# Take first 4 columns ans assign them to variable "X"
X = df.iloc[:,0:4]
# Take first 5th columns and assign them to variable "Y"
Y = df.select_dtypes(include=[object])
X.head()
Y.head()

# Y contains all categories or classes
Y.species.unique()

# Transforming categorial into numerical values
le = preprocessing.LabelEncoder()
Y = Y.apply(le.fit_transform)
Y.head()

# Train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20)

# Feature Scaling
scaler = StandardScaler() 
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,Y_train.values.ravel())
predictions = mlp.predict(X_test)
print(predictions)

# Evaluation of algorithm performance in classifying flowers
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
```
## OUTPUT :
## Dataset Labels :

![279875204-9abac277-edd9-4876-85c6-ceac9e4b520b](https://github.com/charumathiramesh/Experiment-4---Implementation-of-MLP-with-Backpropagation/assets/120204455/5a10fdfa-4450-45f0-bbab-06435a2da2c0)

## X.head() :

![279875275-a245e133-b9d2-493c-90a6-57e220fddd02](https://github.com/charumathiramesh/Experiment-4---Implementation-of-MLP-with-Backpropagation/assets/120204455/21c85ad9-f2dd-4b7e-aac3-ef760e54fedf)



## Y.head() :

![279875369-96bd2136-b07d-47e7-9f28-9f87270a0d66](https://github.com/charumathiramesh/Experiment-4---Implementation-of-MLP-with-Backpropagation/assets/120204455/7218b189-ff1d-44eb-8afc-46712e591221)

## Unique Values in Y :

![279875413-e3b2949b-1575-4e6a-aeaf-4d9be6f8db91](https://github.com/charumathiramesh/Experiment-4---Implementation-of-MLP-with-Backpropagation/assets/120204455/baef6c31-72c1-4b53-8311-f6750c4393e7)

## Predictions :

![279875478-04103951-601c-4d98-87ca-d6da724a3a3d](https://github.com/charumathiramesh/Experiment-4---Implementation-of-MLP-with-Backpropagation/assets/120204455/7f13601e-364e-488d-b0de-bc6fad7f1c70)

## Confusion Matrix :
![279875536-8875a659-effc-4ab0-88ad-0f9db829f199](https://github.com/charumathiramesh/Experiment-4---Implementation-of-MLP-with-Backpropagation/assets/120204455/f8e06843-e805-4b7f-9d3e-33f8fe0555db)


## Classification Report :
![279875584-1cf60252-4cdc-44ff-ae45-b3bffcb5ca2c](https://github.com/charumathiramesh/Experiment-4---Implementation-of-MLP-with-Backpropagation/assets/120204455/612b3c84-540e-43a7-840b-fbc6f491c716)


## RESULT:
Thus, the program to implement Multilayer Perceptron for Multi Classification is successfully implemented.

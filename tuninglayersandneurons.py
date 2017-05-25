#Tuning Layers and Neurons of a neural network
#Because there might be redundancy in the input data, we reduce the hidden layer neurons to 30 instead of 60
#We also standardize the data to improve performance

#Loading important libraries
import numpy
from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#Fixing data reprocibility
seed = 7
numpy.random.seed(seed)

#Loading the dataset
dataframe = read_csv('byron.csv', header=None)#Download sonar dataset from UCI repository
dataset = dataframe.values

#Separating inputs and outputs
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

#Encoding the class outputs
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Creating the model 
def smaller_model():
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
	
#Standardizing the data
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=smaller_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%%, (%.2f%%)" %(results.mean()*100, results.std()*100))

#Halving the network size improved the accuracy and thus halved our training time
#It also reduced the spread ie standard deviation of the data
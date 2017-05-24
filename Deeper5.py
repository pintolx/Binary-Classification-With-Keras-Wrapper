#Binary Classification Using the Keras Library
import numpy
from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Initializing random number generator to ensure reproducibility
seed  = 7
numpy.random.seed(seed)

#Loading the data
dataframe = read_csv('byron.csv', header=None) #Download the data from http://www.is.umk.pl/projects/datasets.html#Sonar
dataset = dataframe.values
#Split the inputs and outputs
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

#Encoding the output to integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Creating the model class
def create_baseline():
    #Create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #Compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
	
#Evaluating the model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" %(results.mean()*100, results.std()*100))
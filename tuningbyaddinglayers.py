#Network Tuning by expanding the model with more layers
import numpy
from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Fixing the seed reproducibility
seed = 7
numpy.random.seed(seed)

#Loading the data
dataframe = read_csv('byron.csv', header=None) #Download the data from http://www.is.umk.pl/projects/datasets.html#Sonar
dataset = dataframe.values

#Separating inputs and outputs
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

#Changing the output to integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Creating the model function
def bigger_model():
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #Compiling the model 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
	
#Standarding the data
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=bigger_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
#Cross Validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print('Standardized: %.2f%%, (%.2f%%)' %(results.mean()*100, resultsstd()*100))
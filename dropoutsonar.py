#Adding Dropout to the visible Layer of the model for the sonar dataset
#Adding dropout helps lift performance of the model
import numpy
from pandas import read_csv
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
from sklearn.pipeline import Pipeline

#Fixing reproducibility
seed = 7
numpy.random.seed(seed)

#Loading the dataset
dataframe = read_csv('byron.csv', header=None) #Download sonar dataset from the UCI REPOSITORY
dataset = dataframe.values

#Splitting Inputs and outputs
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

#Converting output to values
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Defining the baseline model
def baselinem():
    model = Sequential()
    #Adding dropout to visible layer of the model
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid', kernel_constraint=maxnorm(3)))
    #Compiling the model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model
	
#Standardizing the data
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=baselinem, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)

#Cross validating results
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" %(results.mean()*100, results.std()*100))
	
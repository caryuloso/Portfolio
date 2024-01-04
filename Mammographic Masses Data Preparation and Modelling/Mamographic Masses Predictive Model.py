#Carmen So Homework 9

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn import tree

#Read the Mammographic masses dataset 
datasetPath = "/Users/camilleso/Desktop/mammographic_masses.data"
try:
    dataset = pd.read_csv(datasetPath, sep= ",", header = None)
except FileNotFoundError:
    print('The dataset path or file name is wrong. Please input the datasetPath again')
    
#Set column names 
dataset.columns = ['BI-RADS assessment', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
#BI-RADS assessment = non-predicitve attribute and Severity = Target Variable 

#Checking if all the records have only 0s and 1s as inputs for the target variable
targetVarSet = set()
targetVariables = (dataset['Severity'])
for record in targetVariables:
    targetVarSet.add(record)
print('The values of the Target Variable Column are as follows', targetVarSet) #shows that there are records that have 'Severity' as a value for the target variable column (column names were probably mixed up in the middle of the records)

#Remove the rows in the dataframe that have the invalid value of 'Severity'
dataset = dataset[dataset['Severity'].str.contains('Severity') == False]
newTargetVariables = (dataset['Severity'])

#Double Check dataset if the record was removed 
newTargetVarSet = set()
newTargetVar = (dataset['Severity'])
for record in newTargetVar:
    newTargetVarSet.add(record)
print('The values of the Target Variable Column are now fixed. This is the new list of values', newTargetVarSet) #shows that the record was successfully removed - remember: sets store unique values only

#Check if there are missing values and the count of which per column 
def missingNumCount(columnName):
    count = 0
    entries = (dataset[columnName])
    for number in entries:
        if number == '?':
            count = count + 1
    print('The' , columnName, 'Column has' ,count, 'records missing')
            
missingNumCount('Age') #has 5 records with missing ages 
missingNumCount('Shape') #has 31 records with missing shapes
missingNumCount('Margin') #has 48 records with missing margins
missingNumCount('Density') #has 76 records with missing margins 

#based on these findings, check how many records are left if all missing records are removed from the dataset -- if there still is a reasonable number left, we can use this new dataset for ML modelling
features = ['Age', 'Shape', 'Margin', 'Density']
for f in features: 
    dataset = dataset[dataset[f].str.contains('\?') == False]
print('This is the dataset information without the records with missing values', dataset.info()) #there are still 831 entries which is sufficient enough to build a model

#Check if the records have enough benign and malignant records
severityTypes = dataset['Severity']
benignCount = 0
malCount = 0 
for sevtype in severityTypes:
    if sevtype == '0':
        benignCount = benignCount + 1
    else:
        malCount = malCount + 1
print('There are' , benignCount, 'Benign Records') 
print('There are' , malCount, 'Malignant Records') #there is a very even distribution of benign - 428 and malignant - 403 records

#additionally, the values for age are strings instead of integers -- based on the dataset description, these should be integer data types 
dataset['Age'] = dataset['Age'].astype(int)
print('This is the dataset information. it shows the Age values are now integers',dataset.info()) #the values were successfully converted to integers 

#The last column 'BI-RADS assessment' should be removed as to avoid confusion. 
#This is because it is neither a target variable nor a feature with predictive power as indicated on the dataset documentation 
del dataset['BI-RADS assessment']
print('This is the dataset information. it shows the BI-RADS column was successfully removed' , dataset.info()) #the column was successfully removed 

#Reset the indexes as some are missing due to records being removed from the dataframe
dataset = dataset.reset_index(drop= True)

#Based on the Data description, the Shape, Margin and Density features are actually categorical. Because these are represented in numbers, these should be converted to their actual labels instead to avoid confusion
shapeValues = { '1' : 'Round', '2' : 'Oval', '3' : 'Lobular' , '4' : 'Irregular'}
marginValues = {'1' : 'Circumscribed' , '2' : 'Microlobulated' , '3' : 'Obscured', '4' : 'Illdefined' , '5' : 'Spiculated' }
densityValues = {'1' : 'HighDensity' , '2' : 'IsoDensity', '3' : 'LowDensity' , '4' : 'FatContainingDensity'}

#A function to make new rows with the actual value names per column 
def changeValues(colName, colValues):
    i = 0
    while i < len(dataset):
        if dataset[colName][i] == '1':
            dataset.at[i, dataset.columns.get_loc(colName)] = colValues['1']
            
        if dataset[colName][i] == '2':
            dataset.at[i, dataset.columns.get_loc(colName)] = colValues['2']

        if dataset[colName][i] == '3':
            dataset.at[i, dataset.columns.get_loc(colName)] = colValues['3']

        if dataset[colName][i] == '4':
            dataset.at[i, dataset.columns.get_loc(colName)] = colValues['4']
            
        if dataset[colName][i] == '5':
            dataset.at[i, dataset.columns.get_loc(colName)] = 'Spiculated'
        
        i = i + 1

changeValues('Shape', shapeValues)
changeValues('Margin', marginValues)
changeValues('Density', densityValues)

#Drop the old columns with the numeric representatives as we can replace this with the new columns made above
del dataset['Shape']
del dataset['Margin']
del dataset['Density']
dataset.info() #View the new dataset columns

#Get the input variables that need to be binarized 
catInputColumns = [1, 2, 3]
categoricalInputs = dataset[catInputColumns] 

# The next step is to binarize the feature data that are categorical and make a dataframe with these new columns and the age column 
xBinarized = pd.get_dummies(categoricalInputs)
agesColumn = dataset['Age']
inputdf = xBinarized.join(agesColumn)

#Now that the dataframe is ready, the first step is to separate the target and the input variables from the dataframe 
x = inputdf 
y = dataset['Severity']

#Split the data so that the model can use train data to create the model and test data to evaluate the model's performance 
#test data = 20% of the records from the dataset, and random_state = 1 -- if the same  split needs to be referred to again, the random_state number can be referenced
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1) 

#Use data for the model 
model = RandomForestClassifier(n_estimators=2000, criterion='entropy', random_state = 0, max_depth= 3) #Set the criteria for the Random Forest ML model 
model.fit(X_train, Y_train) #Use the model on the training data

Y_Predictions = model.predict(X_test) #Get the predictions for the test data
print('the models accuracy is:' , model.score(X_test, Y_test)) #Get the accuracy of the model - which is 83.23%


#Visualize any of the decision trees used in the random forest - change index of model.estimators_ to see a different tree 
plt.figure(figsize=(20,20))
tree.plot_tree(model.estimators_[1], feature_names=X_test.columns, filled=True)
plt.savefig('randomTreeImage.png', dpi = 300) #Saves it as a png file on your machine 

# shows which features are most important in determining whether a tumour is benign or malignant 
importances = model.feature_importances_ 
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

forest_importances = pd.Series(importances, index=X_test.columns)
forest_importances = forest_importances.sort_values(ascending = False)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax = ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig('featureImportanceFig.png', dpi = 300) #Saves it as a png file on your machine 

#save the dataframe with the new column, predictions made, to a CSV 
totalPredictions = model.predict(x) #makes predictions for the original dataset so the predicitions and the actuals can be plotted on the CSV 
predictionSeries = pd.Series(totalPredictions, name = 'Severity_Predictions')
dfWithPredictions = dataset.join(predictionSeries)
print('This is the new dataframe with a new predicitions column derived by the model' , dfWithPredictions.info())
#change the column names of the categorical features from 1,2, and 3 back to their original names
dfWithPredictions.rename(columns={1: 'Shape', 2: 'Margin', 3: 'Density'}, inplace=True)

dfFileName = '/Users/camilleso/Desktop/SeverityPredictions.csv'

try: 
    with open(dfFileName, 'w', encoding='utf-8') as dataFile:
        dataFile.write('Age, Shape, Margin, Density, Severity_Actuals, Severity_Predictions\n')
        i = 0 
        for age in dfWithPredictions['Age']:
            a = str(age)
            sh = dfWithPredictions['Shape'][i]
            m = dfWithPredictions['Margin'][i]
            d = dfWithPredictions['Density'][i]
            s = str(dfWithPredictions['Severity'][i])
            t = str(dfWithPredictions['Severity_Predictions'][i])
            dataFile.write(a+','+sh+','+m+','+d+','+s+','+t+'\n')
            i = i + 1 #iterate over the indexes in the dataframe 
except FileNotFoundError: 
    print('The filename path is wrong. Please input the path again')



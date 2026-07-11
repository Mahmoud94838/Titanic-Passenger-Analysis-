import pandas as pd

data = { 
    'name': ['Ali', 'Sara', 'Omar'], 
    'age': [20, 22, 19], 
    'score': [85, 90, 78] 
}
# #Create this data frame 
df = pd.DataFrame(data)
# #Print column names 
print(df.columns)
#Print first two rows of "score"
print(df['score'][0:2])
# #Add column named “passed” with values [True, True, False]
# df['passed'] = [True, True, False]
# #Check if age contains value 22 using isin() 
# print(df['age'].isin([22]))
# #Print students with score > 80 “Boolean result” 
# print(df['score'] > 80)
#print students with score between 80 and 90 “values result” 
print(df[(df['score'] >= 80) & (df['score'] <= 90)])

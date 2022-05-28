import numpy as np
import pandas as pd
import numpy as np

df = pd.read_csv('fie.csv')
# #
for row in range(df.shape[0]):
    for col in range(df.shape[1]):
        if not pd.isnull(df.iloc[row,col]):
            print(df.iloc[row, col])



# data = {'set_of_numbers': [1,2,3,4,5,np.nan,6,7,np.nan,8,9,10,np.nan]}
# df = pd.DataFrame(data)

# df.loc[df['set_of_numbers'].isnull(),'value_is_NaN'] = 'Yes'
# df.loc[df['set_of_numbers'].notnull(), 'value_is_NaN'] = 'No'
# for row in range(self.RowNum):
#     for col in range(selfpd.columnNum):
#         if not np.isnan(df.iloc[row, col]):
#             self.tableWidget.setItem(row, col, df.iloc[row, col])
# print (pd.isnull(df.iloc[1,6]))
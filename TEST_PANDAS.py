import numpy as np
import pandas as pd

# group by
a = np.array([[0, 1], [0, 1], [1, 1], [1, 0], [1, 0], [0, 0]])
zeroarray = np.ones((6, 1)) # This array has no meaning but its nothing then group by has no data
g = np.column_stack((a,zeroarray))
# g = np.array([[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 0]])
df = pd.DataFrame(g, columns=["TRUTH", "CLASS", "ONES"])
# df  = pd.DataFrame({'TRUTH':[0,1,1,1,0,1], 'ddd':[1,0,1,0,0,1]})
print(df)
print("df[['TRUTH']].count()->{}".format(df[['TRUTH']].count()))
# print("df.groupby('TRUTH').count()->{}".format(df.groupby('TRUTH').count()))
# print("df.groupby(['TRUTH', 'CLASS']).count()->{}".format(df.groupby(['TRUTH', 'CLASS'])))
dfg = df.groupby(['TRUTH', 'CLASS'], as_index=False)
print(dfg.count())
dfg_array = np.array(dfg.count())
print(dfg_array)
# print(dfg.count().as_matrix)
# print(dfg.count().values)
# df.groupby(("TRUTH"))


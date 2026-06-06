import pandas as pd
x=(2,4,6,8,3,99)
a=(6,8,2,7,4,88)
p={1:[2,4,6,8],2:[8,9,5,6],3:[3,6,9,4]}
y=pd.DataFrame(p)
print(y)
y.to_csv("alpha.csv")
csv
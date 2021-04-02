import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix

truth = ["Camera","Dog","Android","Baby","Keyboard","Dinosaur","Dragon","Bunny","Blackberry","Diet Coke Bottle","Coffee Tin","Car","Mug","Koala","Mug"] 
labels = ["Camera","Dog","Android","Baby","Keyboard","Dinosaur","Dragon","Blackberry","Diet Coke Bottle","Coffee Tin","Car","Mug","Koala","Duck","Bunny"]
pred = ["Camera","Dog","Android","Baby","Camera","Camera","Koala","Koala","Camera","Camera","Android","Camera","Android","Koala","Baby"]

plt.figure(1)
x = confusion_matrix(truth,pred,labels=labels)
df_cm = pd.DataFrame(x, range(15),range(15)); 
sn.heatmap(df_cm, annot=True)
plt.show()
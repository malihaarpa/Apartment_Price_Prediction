import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from google.colab import files


%matplotlib inline
uploaded = files.upload()
data = pd.read_csv("document.csv")

data.head()
data.describe()
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
#sns.despine
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
reg = LinearRegression()
labels = data['price']
# conv_dates = [1 if values == 2014 else 0 for values in data.date ]
# data['date'] = conv_dates
train1 = data.drop(['date', 'price'],axis=1)
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =52)
reg.fit(x_train,y_train)
reg.score(x_test,y_test)
print(100*reg.score(x_test, y_test))
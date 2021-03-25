```python
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
```


```python
X=iris.data
y=iris.target
```


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
```


```python
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
```




    GaussianNB()




```python
y_pred=gnb.predict(X_test)
```


```python
from sklearn import metrics
```


```python
print("gaussian nb model accuracy in %: ",metrics.accuracy_score(y_test,y_pred)*100)
```

    gaussian nb model accuracy in %:  93.33333333333333
    


```python

```

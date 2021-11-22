---
title: "Timeseries analysis with Recurrent neural network (LSTM/GRU)"
image: /images/dreamy.jpg
categories:
  - Regression
tags:
  - content
  - Timeseries forecasting
  - Recurrent neural network
  - LSTM
  - GRU
last_modified_at: 2021-03-17T10:46:49-04:00

---
Timeseries forecasting is one of the ubiquitous task in industry and real life problem. In this Project, I used vapor fraction of boiling dataset for Timeseries forecasting.  

```python
import os
import glob
import pandas as pd
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,Bidirectional,GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from time import time
import itertools
from math import sqrt
import seaborn as sns
```

    Using TensorFlow backend.


# Data import


```python
path = r'/jet/home/mhrahman/Projects/HW5/'
data = pd.read_csv('DS-1_36W_vapor_fraction.txt',sep = '\t')
data = data.rename(columns={'Time (ms)':'Time','Vapor Fraction':'Vapor Fraction'})
v_data  = list(data['Vapor Fraction'])
```


```python
#scaler = MinMaxScaler(feature_range=(0, 1))
#v_data = scaler.fit_transform(data['Vapor Fraction'].values.reshape(-1,1)).flatten()
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Vapor Fraction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.333333</td>
      <td>0.566644</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.666667</td>
      <td>0.564461</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>0.562855</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.333333</td>
      <td>0.565662</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.666667</td>
      <td>0.563902</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4994</th>
      <td>1665.000000</td>
      <td>0.567091</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1665.333333</td>
      <td>0.565522</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>1665.666667</td>
      <td>0.565640</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>1666.000000</td>
      <td>0.565539</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>1666.333333</td>
      <td>0.565258</td>
    </tr>
  </tbody>
</table>
<p>4999 rows × 2 columns</p>
</div>




```python
data.plot.line(x = 'Time', y = 'Vapor Fraction')
plt.savefig(r'/jet/home/mhrahman/Projects/HW5/Figures/Timeseries.jpg',dpi = 300)
plt.show()
```



![png](output_5_0.png)



# Input and output data generation


```python
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Finding the end of the pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # Checking if we are beyond the sequence
        if out_end_ix > len(sequence):
            break

        # Gather input and output parts of pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix : out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
```


```python
steps_in = 50
steps_out = 50
X, Y = split_sequence(v_data,steps_in,steps_out)
```


```python
X = np.reshape(X,(X.shape[0],X.shape[1],1))
X.shape
```




    (4900, 50, 1)



# Train test split


```python
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size= 0.2,shuffle = False)
```

# Model building


```python
model_LSTM = Sequential([
    LSTM(50,input_shape = (x_train.shape[1],x_train.shape[2]),activation = 'relu'),
    Dropout(0.2),
    Dense(steps_out,activation = 'linear')
])
```


```python
model_biLSTM = Sequential([
    Bidirectional(LSTM(50),input_shape = (x_train.shape[1],x_train.shape[2])),
    Dropout(0.2),
    Dense(steps_out)
])
```


```python
model_GRU = Sequential([
    GRU(50, input_shape = (x_train.shape[1],x_train.shape[2])),
    Dropout(0.2),
    Dense(steps_out)
])
```


```python
model_biGRU = Sequential([
    Bidirectional(GRU(50),input_shape = (x_train.shape[1],x_train.shape[2])),
    Dropout(0.2),
    Dense(steps_out)
])
```


```python
model = model_LSTM
model.summary()
with open('modelsummary_LSTM_2.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 50)                10400     
    _________________________________________________________________
    dropout (Dropout)            (None, 50)                0         
    _________________________________________________________________
    dense (Dense)                (None, 50)                2550      
    =================================================================
    Total params: 12,950
    Trainable params: 12,950
    Non-trainable params: 0
    _________________________________________________________________


# Call backs


```python
class TimeCallback(Callback):
    def on_train_begin(self,logs={}):
        self.logs=[]
    def on_epoch_begin(self,epoch,logs={}):
        self.starttime = time()
    def on_epoch_end(self,epoch,logs={}):
        self.logs.append(time()-self.starttime)
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1, patience = 5,min_delta = 1)
cb = TimeCallback()
checkpoints = ModelCheckpoint('weight.hdf5',monitor='loss',verbose=1,save_best_only= True,mode='min')
```

# Model compilation and fitting


```python
model.compile(optimizer='adam', loss = 'mean_squared_error')
```


```python
epochs = 100
batch = 32
t1 = time()
history = model.fit(x_train,y_train,epochs=epochs,
                    batch_size = batch,validation_split= .2,verbose = 1,
                    callbacks = [cb,checkpoints,es],
                   shuffle = False)
t2 = time()
```

    Train on 3136 samples, validate on 784 samples
    Epoch 1/100
    3104/3136 [============================>.] - ETA: 0s - loss: 0.1058
    Epoch 00001: loss improved from inf to 0.10491, saving model to weight.hdf5
    3136/3136 [==============================] - 10s 3ms/sample - loss: 0.1049 - val_loss: 0.0067
    Epoch 2/100
    3104/3136 [============================>.] - ETA: 0s - loss: 0.0106
    Epoch 00002: loss improved from 0.10491 to 0.01058, saving model to weight.hdf5
    3136/3136 [==============================] - 10s 3ms/sample - loss: 0.0106 - val_loss: 0.0036
    Epoch 3/100
    3104/3136 [============================>.] - ETA: 0s - loss: 0.0075
    Epoch 00003: loss improved from 0.01058 to 0.00746, saving model to weight.hdf5
    3136/3136 [==============================] - 8s 3ms/sample - loss: 0.0075 - val_loss: 0.0027
    Epoch 4/100
    3104/3136 [============================>.] - ETA: 0s - loss: 0.0060
    Epoch 00004: loss improved from 0.00746 to 0.00602, saving model to weight.hdf5
    3136/3136 [==============================] - 8s 2ms/sample - loss: 0.0060 - val_loss: 0.0032
    Epoch 5/100
    3104/3136 [============================>.] - ETA: 0s - loss: 0.0052
    Epoch 00005: loss improved from 0.00602 to 0.00516, saving model to weight.hdf5
    3136/3136 [==============================] - 8s 3ms/sample - loss: 0.0052 - val_loss: 0.0029
    Epoch 6/100
    3104/3136 [============================>.] - ETA: 0s - loss: 0.0046
    Epoch 00006: loss improved from 0.00516 to 0.00461, saving model to weight.hdf5
    3136/3136 [==============================] - 10s 3ms/sample - loss: 0.0046 - val_loss: 0.0030
    Epoch 00006: early stopping


# Model Evaluation


```python
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs,loss,'r')
plt.plot(epochs,val_loss,'b')
plt.title('Training and validation loss')
plt.xlabel('epochs',fontsize = 12)
plt.ylabel('Loss', fontsize = 12)
plt.legend(["Training loss","Validation loss"])
plt.savefig(r'/jet/home/mhrahman/Projects/HW5/Figures/Loss.jpg', dpi = 300)
plt.show()
```



![png](output_24_0.png)




```python
plt.plot(cb.logs)
plt.title('Time per epoch')
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.legend(['Time'],loc = 'upper right')
#path = r'/jet/home/mhrahman/Projects/HW1/Figures/Classification_loss.jpg'
plt.savefig(r'/jet/home/mhrahman/Projects/HW5/Figures/Time.jpg', dpi = 300)
plt.show()
```



![png](output_25_0.png)




```python
y_predicted = model.predict(x_test)
error = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
print(error, t2-t1)
```

    0.03582213763699584 54.32372784614563



```python
in_ = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))[0]
Actual = y_test[0]
Predicted = y_predicted[0]
plt.plot(range(0,steps_in),in_,color = 'black',label = 'input singal')
plt.plot(range(steps_in-1,steps_in+steps_out-1),Actual,color = 'red',label = 'Actual singal')
plt.plot(range(steps_in-1,steps_in+steps_out-1),Predicted,color = 'blue',label = 'Predicted signal')
plt.xlabel('Time',fontsize = 14)
plt.ylabel('Vapor fraction',fontsize = 14)
plt.legend()
plt.savefig(r'/jet/home/mhrahman/Projects/HW5/Figures/Signal.jpg',dpi = 300)
plt.show()
```



![png](output_27_0.png)




```python
y_pr = model.predict(x_train)
tr = []
pr = []
for i in range(len(y_train)):
    tr.append(y_train[i][steps_out-1])
    pr.append(y_pr[i][steps_out-1])
plt.figure(figsize=(12,5))
plt.plot(tr,label = "Actual")
plt.plot(range(-50,len(pr)-50),pr,label = "Fifty-step predicted")
plt.legend()
plt.xlabel('Time',fontsize = 14)
plt.ylabel('Vapor fraction',fontsize = 14)
#plt.savefig(r'/jet/home/mhrahman/Projects/HW5/Figures/Total_50.jpg',dpi = 300)
plt.show()
```



![png](output_28_0.png)




```python
tr = []
pr = []
for i in range(len(y_test)):
    tr.append(y_test[i][steps_out-1])
    pr.append(y_predicted[i][steps_out-1])

plt.plot(tr,label = "Actual")
plt.plot(range(-50,len(pr)-50),pr,label = "Fifty-step predicted")
plt.legend()
plt.xlabel('Time',fontsize = 14)
plt.ylabel('Vapor fraction',fontsize = 14)
plt.savefig(r'/jet/home/mhrahman/Projects/HW5/Figures/Total_50.jpg',dpi = 300)
plt.show()
```



![png](output_29_0.png)



# Testing varying input and output length


```python
def error_image(step_in, step_out, data,epochs, batch):
    X,Y = split_sequence(data, step_in, step_out)
    X = np.reshape(X,(X.shape[0],X.shape[1],1))
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size= 0.2,shuffle = False)
    model_LSTM_2 = Sequential([
    LSTM(50,input_shape = (x_train.shape[1],x_train.shape[2])),
    Dropout(0.2),
    Dense(step_out)])
    model = model_LSTM_2
    model.compile(optimizer='adam', loss = 'mean_squared_error')
    model.fit(x_train,y_train,epochs=epochs,
                    batch_size = batch,validation_split= .2,verbose = 1,
                   shuffle = False,callbacks = [es])
    y_predicted = model.predict(x_test)
    error = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
    return error
```


```python
epochs = 15
batch = 32
periods = [25,50,75,100,125,150,175,200]
total_error = []
for i in periods:
    error = []
    for j in periods:
        print("Training for:", i,j)
        er = error_image(i,j,v_data,epochs, batch)
        error.append(er)
    total_error.append(error)
```

    Training for: 25 25
    Train on 3168 samples, validate on 792 samples
    3168/3168 [==============================] - 5s 2ms/sample - loss: 0.1068 - val_loss: 0.0038
    Training for: 25 50
    Train on 3152 samples, validate on 788 samples
    3152/3152 [==============================] - 7s 2ms/sample - loss: 0.1040 - val_loss: 0.0039
    Training for: 25 75
    Train on 3136 samples, validate on 784 samples
    3136/3136 [==============================] - 7s 2ms/sample - loss: 0.0961 - val_loss: 0.0116
    Training for: 25 100
    Train on 3120 samples, validate on 780 samples
    3120/3120 [==============================] - 5s 2ms/sample - loss: 0.0937 - val_loss: 0.0094
    Training for: 25 125
    Train on 3104 samples, validate on 776 samples
    3104/3104 [==============================] - 6s 2ms/sample - loss: 0.0944 - val_loss: 0.0110
    Training for: 25 150
    Train on 3088 samples, validate on 772 samples
    3088/3088 [==============================] - 6s 2ms/sample - loss: 0.0815 - val_loss: 0.0110
    Training for: 25 175
    Train on 3072 samples, validate on 768 samples
    3072/3072 [==============================] - 5s 2ms/sample - loss: 0.0839 - val_loss: 0.0102
    Training for: 25 200
    Train on 3056 samples, validate on 764 samples
    3056/3056 [==============================] - 6s 2ms/sample - loss: 0.1012 - val_loss: 0.0085
    Training for: 50 25
    Train on 3152 samples, validate on 788 samples
    3152/3152 [==============================] - 6s 2ms/sample - loss: 0.0985 - val_loss: 0.0049
    Training for: 50 50
    Train on 3136 samples, validate on 784 samples
    3136/3136 [==============================] - 8s 2ms/sample - loss: 0.0955 - val_loss: 0.0057
    Training for: 50 75
    Train on 3120 samples, validate on 780 samples
    3120/3120 [==============================] - 8s 3ms/sample - loss: 0.1124 - val_loss: 0.0072
    Training for: 50 100
    Train on 3104 samples, validate on 776 samples
    3104/3104 [==============================] - 6s 2ms/sample - loss: 0.1139 - val_loss: 0.0104
    Training for: 50 125
    Train on 3088 samples, validate on 772 samples
    3088/3088 [==============================] - 7s 2ms/sample - loss: 0.0903 - val_loss: 0.0094
    Training for: 50 150
    Train on 3072 samples, validate on 768 samples
    3072/3072 [==============================] - 6s 2ms/sample - loss: 0.1048 - val_loss: 0.0115
    Training for: 50 175
    Train on 3056 samples, validate on 764 samples
    3056/3056 [==============================] - 7s 2ms/sample - loss: 0.1288 - val_loss: 0.0188
    Training for: 50 200
    Train on 3040 samples, validate on 760 samples
    3040/3040 [==============================] - 7s 2ms/sample - loss: 0.1050 - val_loss: 0.0204
    Training for: 75 25
    Train on 3136 samples, validate on 784 samples
    3136/3136 [==============================] - 7s 2ms/sample - loss: 0.0973 - val_loss: 0.0093
    Training for: 75 50
    Train on 3120 samples, validate on 780 samples
    3120/3120 [==============================] - 9s 3ms/sample - loss: 0.1041 - val_loss: 0.0090
    Training for: 75 75
    Train on 3104 samples, validate on 776 samples
    3104/3104 [==============================] - 8s 3ms/sample - loss: 0.1106 - val_loss: 0.0162
    Training for: 75 100
    Train on 3088 samples, validate on 772 samples
    3088/3088 [==============================] - 7s 2ms/sample - loss: 0.1137 - val_loss: 0.0100
    Training for: 75 125
    Train on 3072 samples, validate on 768 samples
    3072/3072 [==============================] - 7s 2ms/sample - loss: 0.1651 - val_loss: 0.0704
    Training for: 75 150
    Train on 3056 samples, validate on 764 samples
    3056/3056 [==============================] - 7s 2ms/sample - loss: 0.1031 - val_loss: 0.0105
    Training for: 75 175
    Train on 3040 samples, validate on 760 samples
    3040/3040 [==============================] - 7s 2ms/sample - loss: 0.3534 - val_loss: 0.1221
    Training for: 75 200
    Train on 3024 samples, validate on 756 samples
    3024/3024 [==============================] - 8s 3ms/sample - loss: 0.1578 - val_loss: 0.0669
    Training for: 100 25
    Train on 3120 samples, validate on 780 samples
    3120/3120 [==============================] - 7s 2ms/sample - loss: 0.1334 - val_loss: 0.0430
    Training for: 100 50
    Train on 3104 samples, validate on 776 samples
    3104/3104 [==============================] - 9s 3ms/sample - loss: 0.1155 - val_loss: 0.0097
    Training for: 100 75
    Train on 3088 samples, validate on 772 samples
    3088/3088 [==============================] - 10s 3ms/sample - loss: 101.9191 - val_loss: 0.1466
    Training for: 100 100
    Train on 3072 samples, validate on 768 samples
    3072/3072 [==============================] - 7s 2ms/sample - loss: 6.8368 - val_loss: 0.1498
    Training for: 100 125
    Train on 3056 samples, validate on 764 samples
    3056/3056 [==============================] - 8s 2ms/sample - loss: 0.1681 - val_loss: 0.0703
    Training for: 100 150
    Train on 3040 samples, validate on 760 samples
    3040/3040 [==============================] - 8s 2ms/sample - loss: 59.6738 - val_loss: 0.1325
    Training for: 100 175
    Train on 3024 samples, validate on 756 samples
    3024/3024 [==============================] - 8s 2ms/sample - loss: 866559.0253 - val_loss: 0.1612
    Training for: 100 200
    Train on 3008 samples, validate on 752 samples
    3008/3008 [==============================] - 7s 2ms/sample - loss: 0.1050 - val_loss: 0.0133
    Training for: 125 25
    Train on 3104 samples, validate on 776 samples
    3104/3104 [==============================] - 8s 2ms/sample - loss: 0.1046 - val_loss: 0.0088
    Training for: 125 50
    Train on 3088 samples, validate on 772 samples
    3088/3088 [==============================] - 9s 3ms/sample - loss: 0.1195 - val_loss: 0.0151
    Training for: 125 75
    Train on 3072 samples, validate on 768 samples
    3072/3072 [==============================] - 10s 3ms/sample - loss: 0.1036 - val_loss: 0.0072
    Training for: 125 100
    Train on 3056 samples, validate on 764 samples
    3056/3056 [==============================] - 8s 3ms/sample - loss: 31.7545 - val_loss: 0.1133
    Training for: 125 125
    Train on 3040 samples, validate on 760 samples
    3040/3040 [==============================] - 9s 3ms/sample - loss: 28395.2216 - val_loss: 0.1458
    Training for: 125 150
    Train on 3024 samples, validate on 756 samples
    3024/3024 [==============================] - 8s 3ms/sample - loss: 0.0913 - val_loss: 0.0079
    Training for: 125 175
    Train on 3008 samples, validate on 752 samples
    3008/3008 [==============================] - 9s 3ms/sample - loss: 0.1114 - val_loss: 0.0134
    Training for: 125 200
    Train on 2992 samples, validate on 748 samples
    2992/2992 [==============================] - 10s 3ms/sample - loss: 445529251.9565 - val_loss: 0.1636
    Training for: 150 25
    Train on 3088 samples, validate on 772 samples
    3088/3088 [==============================] - 10s 3ms/sample - loss: 0.1144 - val_loss: 0.0082
    Training for: 150 50
    Train on 3072 samples, validate on 768 samples
    3072/3072 [==============================] - 10s 3ms/sample - loss: nan - val_loss: nan



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-91-e08b93846b33> in <module>
          7     for j in periods:
          8         print("Training for:", i,j)
    ----> 9         er = error_image(i,j,v_data,epochs, batch)
         10         error.append(er)
         11     total_error.append(error)


    <ipython-input-86-41d606cb5cee> in error_image(step_in, step_out, data, epochs, batch)
         13                    shuffle = False)
         14     y_predicted = model.predict(x_test)
    ---> 15     error = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
         16     return error


    ~/ML/Project/lib/python3.7/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74


    ~/ML/Project/lib/python3.7/site-packages/sklearn/metrics/_regression.py in mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)
        254     """
        255     y_type, y_true, y_pred, multioutput = _check_reg_targets(
    --> 256         y_true, y_pred, multioutput)
        257     check_consistent_length(y_true, y_pred, sample_weight)
        258     output_errors = np.average((y_true - y_pred) ** 2, axis=0,


    ~/ML/Project/lib/python3.7/site-packages/sklearn/metrics/_regression.py in _check_reg_targets(y_true, y_pred, multioutput, dtype)
         84     check_consistent_length(y_true, y_pred)
         85     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    ---> 86     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
         87
         88     if y_true.ndim == 1:


    ~/ML/Project/lib/python3.7/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74


    ~/ML/Project/lib/python3.7/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        643         if force_all_finite:
        644             _assert_all_finite(array,
    --> 645                                allow_nan=force_all_finite == 'allow-nan')
        646
        647     if ensure_min_samples > 0:


    ~/ML/Project/lib/python3.7/site-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype)
         97                     msg_err.format
         98                     (type_err,
    ---> 99                      msg_dtype if msg_dtype is not None else X.dtype)
        100             )
        101     # for object dtype data, we only check for NaNs (GH-13254)


    ValueError: Input contains NaN, infinity or a value too large for dtype('float32').



```python
t = np.array(total_error)
np.save("total.npy",t)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-20-b5c805b1b6aa> in <module>
    ----> 1 t = np.array(total_error)
          2 np.save("total.npy",t)


    NameError: name 'total_error' is not defined


# Ploting the RMSE as heatmap


```python
periods = [25,50,75,100,125,150,175,200]
df = pd.DataFrame(np.load('total.npy'),columns = periods,index = periods)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>25</th>
      <th>50</th>
      <th>75</th>
      <th>100</th>
      <th>125</th>
      <th>150</th>
      <th>175</th>
      <th>200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0.023868</td>
      <td>0.033878</td>
      <td>0.040942</td>
      <td>0.045768</td>
      <td>0.047754</td>
      <td>0.050334</td>
      <td>0.052212</td>
      <td>0.053568</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.024400</td>
      <td>0.033632</td>
      <td>0.040296</td>
      <td>0.047425</td>
      <td>0.049708</td>
      <td>0.051926</td>
      <td>0.054857</td>
      <td>0.054075</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.024789</td>
      <td>0.034504</td>
      <td>0.040360</td>
      <td>0.044982</td>
      <td>0.049900</td>
      <td>0.050633</td>
      <td>0.054703</td>
      <td>0.054784</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.024641</td>
      <td>0.033854</td>
      <td>0.041248</td>
      <td>0.045054</td>
      <td>0.049316</td>
      <td>0.049669</td>
      <td>0.052954</td>
      <td>0.053442</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0.025744</td>
      <td>0.033843</td>
      <td>0.041957</td>
      <td>0.044949</td>
      <td>0.048502</td>
      <td>0.051703</td>
      <td>0.053974</td>
      <td>0.056907</td>
    </tr>
    <tr>
      <th>150</th>
      <td>0.030218</td>
      <td>0.033840</td>
      <td>0.043066</td>
      <td>0.045135</td>
      <td>0.049259</td>
      <td>0.050350</td>
      <td>0.054937</td>
      <td>0.055030</td>
    </tr>
    <tr>
      <th>175</th>
      <td>0.028828</td>
      <td>0.035109</td>
      <td>0.041617</td>
      <td>0.049458</td>
      <td>0.049301</td>
      <td>0.054526</td>
      <td>0.053646</td>
      <td>0.055493</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.030701</td>
      <td>0.034253</td>
      <td>0.045742</td>
      <td>0.046048</td>
      <td>0.050306</td>
      <td>0.052639</td>
      <td>0.054813</td>
      <td>0.058196</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,8))
sns.heatmap(df,annot=True)
plt.xlabel("Output vector length", fontsize = 14)
plt.ylabel("Input vector length", fontsize = 14)
plt.savefig(r'/jet/home/mhrahman/Projects/HW5/Figures/Heatmap.jpg',dpi = 300)
plt.show()
```



![png](output_36_0.png)

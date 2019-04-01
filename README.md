# DSAI
- [View on jupyter notebooks](https://nbviewer.jupyter.org/github/jeremywen0202/DSAI_HW1/blob/master/forecasting.ipynb)
###### tags: `DSAI`
## HW1
- 温承達
- P76071145
## Environment
- Ubuntu 18.04.1 LTS (GNU/Linux 4.15.0-34-generic x86_64)
- Python 3.6.7
## Install Dependency
- pip install -r requirements.txt
## Files Structure

```
.
+-- data
|   +-- 2017.csv
|   +-- 2018-19.csv
+-- next_week_data
|   +-- next_week.csv
+-- requirements.txt
+-- submission.csv
+-- app.py
```
- `data/`: 台電過去電力供需資訊(從2017/01/01 - 2019/02/28)
- `next_week_data/next_week.csv`: 台電未來一週電力供需預測
- `requirements.txt`: python套件需求
- `submission.csv`: 作業規定之'電力尖峰負載'預測檔案
- `app.py`: 主程式

## Data Exploration
- 我使用[台電公佈的過去電力供需資訊](https://data.gov.tw/dataset/19995)，有兩個檔案，一個包含2017-2018年的資料，一個包含2018-2019年的資料，由於2018年有重疊，因此我將其中一個2018年份資訊刪掉，得到`Data/2017.csv` 和 `Data/2018-19.csv` 兩個檔案，之後我發現資料欄位裡的中文字會造成在server上產生亂碼，導致無法執行程式，因此我將資料集裡的中文都先改成英文。

## Feature Selection
- 在台電公佈的電力資訊中，我認為`淨尖峰供電能力(MW)`應該會是一個很準確的feature，因為這是台電根據自身發電機組狀況、天氣、各種環境條件下，所評估出的應具備供電能力，所以，如果拿它來當training data應該會很準確。

- 因此我將`Data/2018-19.csv`和`Data/2018-19.csv`中的`Power_Supply(MW)`欄位當作x_train、`Peak_Load(MW)`當作y_train。

## Result
- 將`Power_Supply(MW)`都讀進x_train、`Peak_Load(MW)`都讀進y_train後，我將它們轉成np.array，並使用`next_week_data/next_week.csv`([台電未來一週電力預測](https://data.gov.tw/dataset/33462))中的'預估淨尖峰供電能力(萬瓩)'當作x_test(在程式碼中我命名為x_next)，並將'預估瞬時尖峰負載(萬瓩)'當作y_test(在程式碼中我命名為y_next)以進行驗證RMSE。


- 接下來我嘗試了Linear Regression、SVM model中的各種kernel，發現若使用SVM model，kernel設為'poly'，且degree設為9時，平均的RMSE表現會最好，因此我最後是使用SVM model 參數設為：kernel='poly', max_iter=20000, gamma='scale', C=1.0, degree=9。

## My Code
```python=
def warn(*args, **kwargs):    
    pass
import warnings
warnings.warn = warn

if __name__ == '__main__':
    import numpy as np
    import csv 
    import os.path
    from sklearn import svm 
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    path = './data/'
    x_train = []
    y_train = []
    
    ##create x_train and y_train
    for _, _, filenames in os.walk(path):  
        for filename in filenames:
            with open(path + filename, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
    
                for row in rows:
                    x_train.append(float(row['Power_Supply(MW)']))
                    y_train.append(float(row['Peak_Load(MW)']))

    ##transform to np.array data type
    x_train = np.array(x_train)    
    y_train = np.array(y_train)
    _x_train = x_train.reshape(-1, 1)
    _y_train = y_train.reshape(-1, 1)

    ##use SVM model to train data
    clf = svm.SVC(kernel='poly', max_iter=20000, gamma='scale', C=1.0, degree=9)
    clf.fit(_x_train, _y_train)

    ##use ./next_week_data/next_week.csv data to predict
    x_next = np.array([30560, 30560, 28140, 27060, 26970, 26930, 30600])
    y_next = np.array([28700, 28600, 25700, 24600, 24300, 24500, 28500])
    predict = clf.predict(x_next.reshape(-1, 1)) 

    ##write results to submission.csv
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'peak_load(MW)'])
        tmp = 0 
        for i in range(402, 409):
            writer.writerow( ['20190' + str(i), int(predict[tmp]) ])
            tmp += 1

    print('My predict of next week: ', predict)
    print('         label:          ', y_next)
    print('         RMSE:           ', sqrt(mean_squared_error(y_next, predict)))

```


- 以下附上我的測試結果：

```
My predict of next week peak load(MW):
 [28631. 28631. 25377. 24722. 24509. 24617. 28602.]

Label:
 [28700 28600 25700 24600 24300 24500 28500]

RMSE:
 165.92123776918115

```

## Authors
[Cheng-Da Wen](https://github.com/jeremywen0202)

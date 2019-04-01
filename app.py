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

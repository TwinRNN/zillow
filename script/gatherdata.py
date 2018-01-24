import dataset_zillow
import numpy as np
def gather():
    dataset = dataset_zillow(128,20,time_series_params)
    dataset.split(3600,117600,3600,3600,0)
    total_batch = int(150000/128)


    X_train=None
    Y_train=None
    macro_train = None
    for i in range(total_batch):
        train, tr_logerror, macro_features_days, macro_features_weeks, macro_features_months = dataset.next_batch()
        if X_train is None:
            X_train = train
        else:
            np.concatenate(X_train,train,axis=0)
        if Y_train is None:
            Y_train = tr_logerror
        else:
            np.concatenate(Y_train,tr_logerror,axis=0)
        if macro_train is None:
            macro_train = macro_days
        else:
            np.concatenate(macro_train,macro_features_days,axis=0)

    valid, valid_logerror, macro_features_days, macro_features_weeks, macro_features_months= dataset.valid_data()

    X_valid = valid
    Y_valid = valid_logerror
    macro_valid = macro_features_days

    test, test_logerror, macro_features_days, macro_features_weeks, macro_features_months = dataset.test_data()
    X_test = test
    Y_test= test_logerror
    macro_test=macro_features_days

    return X_train,Y_train,macro_train,X_valid,Y_valid,macro_valid,X_test,Y_test,macro_test

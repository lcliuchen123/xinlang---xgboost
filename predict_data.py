
from model import *
from process_data import *

def get_result(xgb_params, single_xtrain, single_ytrain, single_xtest, single_ytest):
    #     bst = XGBoostRegressor(num_boost_round=150, eta=0.05, gamma=0.2, max_depth=5, min_child_weight=3,
    #                                     colsample_bytree=0.5, subsample=0.5)
    bst = XGBoostRegressor(**xgb_params)
    bst.fit(single_xtrain, single_ytrain)
    ybeta = bst.predict(single_xtest)
    test_mae = mae_score(ybeta, single_ytest)
    print("mae: ", test_mae)

    return test_mae, ybeta
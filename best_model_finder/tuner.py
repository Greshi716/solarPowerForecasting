from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics  import roc_auc_score,accuracy_score,r2_score
from rocaucreg import regression_roc_auc_score
from application_logging.logger import App_Logger

class Model_Finder:
    '''
    This class is used to find the model with best accuracy and AUC score
    '''
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
        self.reg=RandomForestRegressor()
        self.xgb=XGBRegressor()

    def get_best_params_for_random_forest(self,train_x,train_y):
        '''
        this method will get the parameters for Random forest algorithm which give the best accuracy.
        '''
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            self.param_grid={"n_estimators": [90, 100, 115, 130], "criterion": ["squared_error", "absolute_error", "poisson"],
                               "max_depth": range(2, 20, 1), "max_features": ["sqrt", "log2"],"random_state":[50,60,30,90,100,500]}
            # self.grid=GridSearchCV(estimator=self.reg,param_grid=self.param_grid,cv=5,verbose=3)
            # self.grid.fit(train_x,train_y)
            # self.criterion=self.grid.best_params_['criterion']
            # self.max_depth= self.grid.best_params_['max_depth']
            # self.max_features = self.grid.best_params_['max_features']
            # self.n_estimators = self.grid.best_params_['n_estimators']
            # self.random_state = self.grid.best_params_['random_state']

            # self.reg=RandomForestRegressor(n_estimators=self.n_estimators,criterion=self.criterion,
            #                                   max_depth=self.max_depth, max_features=self.max_features,random_state=self.random_state)
            self.reg=RandomForestRegressor(n_estimators=130,criterion='absolute_error',
                                              max_depth=13, max_features='sqrt',random_state=50)
            self.reg.fit(train_x,train_y)
            # self.logger_object.log(self.file_object, 'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            return self.reg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self, train_x, train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {
                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            # self.grid = GridSearchCV(XGBRegressor(), self.param_grid_xgboost, verbose=3,
            #                          cv=5)
            # finding the best parameters
            # self.grid.fit(train_x, train_y)

            # extracting the best parameters
            # self.learning_rate = self.grid.best_params_['learning_rate']
            # self.max_depth = self.grid.best_params_['max_depth']
            # self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            # self.xgb = XGBRegressor(learning_rate=self.learning_rate, max_depth=self.max_depth,
            #                          n_estimators=self.n_estimators)
            self.xgb = XGBRegressor(learning_rate=0.1, max_depth=3,
                                     n_estimators=50)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            # self.logger_object.log(self.file_object,
            #                        'XGBoost best params: ' + str(
            #                            self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x,train_y,test_x,test_y):
        '''Finds out the model with best AUC score'''
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        try:
            self.xgboost=self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost=self.xgboost.predict(test_x)
            if len(test_y.unique())==1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score=accuracy_score(test_y,self.prediction_xgboost)
                self.logger_object.log(self.file_object,'Accuracy for XGBoost:' +str(self.xgboost_score))
            else:
                self.xgboost_score=regression_roc_auc_score(test_y,self.prediction_xgboost,num_rounds='exact')
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score))

            self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_x)  # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y, self.prediction_random_forest)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = regression_roc_auc_score(test_y, self.prediction_random_forest,'exact')  # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.random_forest_score))
                
            self.r2score_rf=r2_score(test_y,self.prediction_random_forest)
            self.r2score_xgb=r2_score(test_y,self.prediction_xgboost)
            if(self.random_forest_score<self.xgboost_score):
                return 'XGBoost',self.xgboost,self.r2score_xgb
            else:
                return 'RandomForest',self.random_forest,self.r2score_rf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

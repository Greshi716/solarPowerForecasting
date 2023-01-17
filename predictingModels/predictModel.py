import trainingModels
from application_logging.logger import App_Logger
from file_operations.file_methods import File_Operation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor



class predictModel1:
    def __init__(self):
        self.file_object=open("Prediction_Logs/Prediction_Log.txt",'a+')
        self.log_writer=App_Logger()
    
    def predictionFromModel(self,temperature,pressure,humidity,zeinth,azimuth,bestmodel):
        print("temperature:")
        print(temperature)
        print("pressure:")
        print(pressure)
        print("humidity:")
        print(humidity)
        print("zeinth:")
        print(zeinth)
        print("azimuth:")
        print(azimuth)
        print(bestmodel)
        file_op=File_Operation(self.file_object,self.log_writer)
        load_model=file_op.load_model(bestmodel)
        if(bestmodel=='XGBoost'):
            prediction=load_model.predict_proba([[temperature, pressure, humidity, zeinth,azimuth]])
            print(f"Prediction is: { prediction } Kw")
        else:
            prediction=load_model.predict([[temperature, pressure, humidity, zeinth,azimuth]])
            print(f"Prediction is: { prediction } Kw")
        
        return prediction
  
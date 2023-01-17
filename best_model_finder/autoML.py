from application_logging.logger import App_Logger
import evalml
from evalml.preprocessing import split_data
from evalml.automl import AutoMLSearch
class automl:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object

    def bestmodel(self,modelx,modely):
        self.logger_object.log(self.file_object,'Start of AutoMl')
        try:
            xtrain,xtest,ytrain,ytest=split_data(modelx,modely,problem_type='regression')
            best=AutoMLSearch(xtrain,ytrain,problem_type='regression')
            best.search()
            pipeline_name=best.rankings[0:1]['pipeline_name']
            validationscore=best.rankings[0:1]['validation_score']
            meancvscore=best.rankings[0:1]['mean_cv_score']
            percentagebetterthanbaseline=best.rankings[0:1]['percent_better_than_baseline']
            self.logger_object.log(self.file_object,
                                   'Best pipeline found!. Exited the bestmodel method of the automl class')
            print(pipeline_name)
            print(validationscore)
            print(meancvscore)
            print(percentagebetterthanbaseline)
            self.logger_object.log(self.file_object,str(pipeline_name))
            self.logger_object.log(self.file_object,str(validationscore))
            self.logger_object.log(self.file_object,str(meancvscore))
            self.logger_object.log(self.file_object,str(percentagebetterthanbaseline))
          
            return pipeline_name,validationscore,meancvscore,percentagebetterthanbaseline

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in bestmodel method of the automl class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding best pipeline failed. Exited the bestmodel method of the automl class')
            raise Exception()        

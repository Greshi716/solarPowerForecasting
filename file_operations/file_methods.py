import pickle
import os
import shutil
class File_Operation:
    '''
    This class will be used to save the model after training
    '''
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
        self.model_directory='models/'

    def save_model(self,model,filename):
        self.logger_object.log(self.file_object,'Entered the save_model method of the File_operation class')
        try:
            path=os.path.join(self.model_directory,filename)
            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path+'/'+filename+'.sav','wb') as f:
                pickle.dump(model,f)
            self.logger_object.log(self.file_object,'Model File'+filename+' saved. Exited the save_model method of the Model_Finder class')
            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class')
            raise Exception() 
    
    def load_model(self,filename):
        ''' load the model file to memory'''
        self.logger_object.log(self.file_object,'Entered the load_model method of the File_Operation class')
        try:
            with open(self.model_directory+filename+'/'+filename+'.sav','rb') as f:
                self.logger_object.log(self.file_object,
                                       'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
                return pickle.load(f)
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()
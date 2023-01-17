import pandas as pd
import numpy as np
class Data_Getter:
    '''
    This class is used for obtaining the data from the source for training.
    '''
    def __init__(self,file_object,logger_object):
        self.fileobject=file_object
        self.logger_object=logger_object

    def split_data(self):
        '''
        This method splits the dataset according to temperature range
        '''
        self.logger_object.log(self.fileobject,'Entered the get_data method of the Data_Getter class')
        df=pd.read_csv("C:/Users/Asus/Downloads/spg.csv")
   
        df['LabeledTemp'] = pd.cut(x=df['temperature_2_m_above_gnd'], bins=[-5.35 , 10, 25 ,34],
					labels=['cold', 'moderate', 'sunny',])
        model1=df[df['LabeledTemp'] =='cold']
        model2=df[df['LabeledTemp'] =='moderate']
        model3=df[df['LabeledTemp'] =='sunny']
        return model1,model2,model3
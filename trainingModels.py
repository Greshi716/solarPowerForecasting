from sklearn.model_selection import train_test_split
from data_ingestion.data_loader import Data_Getter
from data_preprocessing.preprocessing import Preprocessor
from best_model_finder.tuner import Model_Finder
from best_model_finder.autoML import automl
from application_logging.logger import App_Logger
from file_operations.file_methods import File_Operation
from datavisulazing.datavisulazer import datavisulazer
import tempextraction 

class trainModel:
    def __init__(self):
        self.log_writer=App_Logger()
        self.file_object=open("Training_Logs/ModelTrainingLog.txt",'a+')

    def trainingModels(self,city):
        self.log_writer.log(self.file_object,'Start of Training')
        try:
            data_getter=Data_Getter(self.file_object,self.log_writer)
            model1,model2,model3=data_getter.split_data() 
            
            model1.drop(columns=['LabeledTemp','wind_speed_10_m_above_gnd','wind_speed_80_m_above_gnd','total_precipitation_sfc','snowfall_amount_sfc','total_cloud_cover_sfc','high_cloud_cover_high_cld_lay','medium_cloud_cover_mid_cld_lay','low_cloud_cover_low_cld_lay','shortwave_radiation_backwards_sfc','wind_direction_10_m_above_gnd','wind_direction_80_m_above_gnd','wind_speed_900_mb','wind_direction_900_mb','wind_gust_10_m_above_gnd','angle_of_incidence'],inplace=True)
            model2.drop(columns=['LabeledTemp','wind_speed_10_m_above_gnd','wind_speed_80_m_above_gnd','total_precipitation_sfc','snowfall_amount_sfc','total_cloud_cover_sfc','high_cloud_cover_high_cld_lay','medium_cloud_cover_mid_cld_lay','low_cloud_cover_low_cld_lay','shortwave_radiation_backwards_sfc','wind_direction_10_m_above_gnd','wind_direction_80_m_above_gnd','wind_speed_900_mb','wind_direction_900_mb','wind_gust_10_m_above_gnd','angle_of_incidence'],inplace=True)
            model3.drop(columns=['LabeledTemp','wind_speed_10_m_above_gnd','wind_speed_80_m_above_gnd','total_precipitation_sfc','snowfall_amount_sfc','total_cloud_cover_sfc','high_cloud_cover_high_cld_lay','medium_cloud_cover_mid_cld_lay','low_cloud_cover_low_cld_lay','shortwave_radiation_backwards_sfc','wind_direction_10_m_above_gnd','wind_direction_80_m_above_gnd','wind_speed_900_mb','wind_direction_900_mb','wind_gust_10_m_above_gnd','angle_of_incidence'],inplace=True)

            temperature=tempextraction.temperature()
            city=city
            temperature,pressure,humidity=temperature.temp(city)
            print("Temperature is:")
            temperature=float(temperature)
            temperature=temperature-273.15
            print(temperature)
            if(temperature>-5.35 and  temperature<10):
                self.log_writer.log(self.file_object,'Entered model1 i.e cold data')

                model1vis=datavisulazer()
                shape=model1vis.shape(model1)
                print("shape of model is : ")
                print(shape)
                des=model1vis.describe(model1)
                print("description:")
                print(des)
                model1vis.histogram(model1)
                model1vis.heatmap(model1)
                model1vis.boxplot(model1)

                model1preprocessor=Preprocessor(self.file_object,self.log_writer)
                is_null_present,columns=model1preprocessor.check_for_missingvalues(model1)
                if(is_null_present):
                    model1=model1preprocessor.filling_missing_values(model1,columns=columns)

                
                model1_azimuth,model1_zeinth=model1preprocessor.get_maxoccurance(model1)
                model1_x,model1_y=model1preprocessor.seperate_features_target(model1,label_column_name='generated_power_kw')
                model1_x=model1preprocessor.features_selection(model1_x)

                x1_train,x1_test,y1_train,y1_test=train_test_split(model1_x,model1_y)

                # model1_azimuth,model1_zeinth=model1preprocessor.get_maxoccurance(model1)
                model1_bestmodel=automl(self.file_object,self.log_writer)
                pipeline_name,validationscore,meancvscore,percentagebetterthanbaseline=model1_bestmodel.bestmodel(model1_x,model1_y)

                model1_finder=Model_Finder(self.file_object,self.log_writer)
                best_model1_name,best_model1,best_r2_score1=model1_finder.get_best_model(x1_train,y1_train,x1_test,y1_test)
                file_op=File_Operation(self.file_object,self.log_writer)
                save_model1=file_op.save_model(best_model1,best_model1_name)
                score_model1=best_r2_score1
                self.log_writer.log(self.file_object,'Successful End Of Training')
                self.file_object.close()
                return temperature,pressure,humidity,model1_zeinth,model1_azimuth,best_model1_name

                
            elif(temperature>=10 and temperature<25):
                self.log_writer.log(self.file_object,'Entered model2 i.e moderate data') 
                model2vis=datavisulazer()
                shape=model2vis.shape(model2)
                print("shape of model is : ")
                print(shape)
                des=model2vis.describe(model2)
                print("description:")
                print(des)
                model2vis.histogram(model2)
                model2vis.heatmap(model2)
                model2vis.heatmap(model2)

                model2preprocessor=Preprocessor(self.file_object,self.log_writer)
                is_null_present,columns=model2preprocessor.check_for_missingvalues(model2)
                if(is_null_present):
                    model2=model2preprocessor.filling_missing_values(model2,columns=columns)

               
                model2_azimuth,model2_zeinth=model2preprocessor.get_maxoccurance(model2)
                model2_x,model2_y=model2preprocessor.seperate_features_target(model2,label_column_name='generated_power_kw')
                model2_x=model2preprocessor.features_selection(model2_x)
                x2_train,x2_test,y2_train,y2_test=train_test_split(model2_x,model2_y)
                
                model2_azimuth,model2_zeinth=model2preprocessor.get_maxoccurance(model2)
                model2_bestmodel=automl(self.file_object,self.log_writer)
                pipeline_name,validationscore,meancvscore,percentagebetterthanbaseline=model2_bestmodel.bestmodel(model2_x,model2_y)


                model2_finder=Model_Finder(self.file_object,self.log_writer)
                best_model2_name,best_model2,best_r2_score2=model2_finder.get_best_model(x2_train,y2_train,x2_test,y2_test)
                file_op=File_Operation(self.file_object,self.log_writer)
                save_model2=file_op.save_model(best_model2,best_model2_name)
                score_model2=best_r2_score2
                self.log_writer.log(self.file_object,'Successful End Of Training')
                self.file_object.close()
                return temperature,pressure,humidity,model2_zeinth,model2_azimuth,best_model2_name

 
            else:
                self.log_writer.log(self.file_object,'Entered model3 i.e sunny data')
                
                model3vis=datavisulazer()
                shape=model3vis.shape(model3)
                print("shape of model is : ")
                print(shape)
                des=model3vis.describe(model3)
                print("description:")
                print(des)
                # print(model3.columns)
                model3vis.histogram(model3)
                model3vis.heatmap(model3)
                model3vis.heatmap(model3)

                model3preprocessor=Preprocessor(self.file_object,self.log_writer)
                is_null_present,columns=model3preprocessor.check_for_missingvalues(model3)
                if(is_null_present):
                    model3=model3preprocessor.filling_missing_values(model3,columns=columns)
               

                model3_azimuth,model3_zeinth=model3preprocessor.get_maxoccurance(model3)
                model3_x,model3_y=model3preprocessor.seperate_features_target(model3,'generated_power_kw')
                model3_x=model3preprocessor.features_selection(model3_x)

                x3_train,x3_test,y3_train,y3_test=train_test_split(model3_x,model3_y)

                model3_azimuth,model3_zeinth=model3preprocessor.get_maxoccurance(model3)
                model3_bestmodel=automl(self.file_object,self.log_writer)
                pipeline_name,validationscore,meancvscore,percentagebetterthanbaseline=model3_bestmodel.bestmodel(model3_x,model3_y)

                model3_finder=Model_Finder(self.file_object,self.log_writer)
                best_model3_name,best_model3,best_r2_score3=model3_finder.get_best_model(x3_train,y3_train,x3_test,y3_test)
                file_op=File_Operation(self.file_object,self.log_writer)
                save_model3=file_op.save_model(best_model3,best_model3_name)
                score_model3=best_r2_score3
                self.log_writer.log(self.file_object,'Successful End Of Training')
                self.file_object.close()
                return temperature,pressure,humidity,model3_zeinth,model3_azimuth,best_model3_name


        except Exception:
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception

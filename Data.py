import numpy as np
import pandas as pd
import cv2
import time
import os
from Keys import Keys
from collections import Counter
from random import shuffle
from Agent import Agent
from Screen import Screen
from CNN_Chintan_V2_With_Barrier import CNN_V2_With_Barrier
from CNN_Only_Goalkeeper import CNN_Only_Goalkeeper
from CNN_Fk_With_Goalkeeper_All import CNN_Fk_With_Goalkeeper_All

class Data():

    def __init__(self, file_name='training_data.npy'):
        self.file_name = file_name
        self.path_file = 'Data/' + file_name
        self.training_data = self.InitializeTrainingData()  

	# Load training data if exists, else return an empty list
    def InitializeTrainingData(self):
        if os.path.isfile(self.path_file):
            print('File exists, loading previous data!')
            return list(np.load(self.path_file))
        print('File does not exist, creating file!')
        return []
	
	# Show all data in training data (image, output_move, output_action)
    def validate_data(self):
        for data in self.training_data:
            img = data[0]
            output_move = data[1]
            cv2.imshow('test', img)
            print(output_move)
            if cv2.waitKey(25) & 0xFF == ord('q'): # Destroy all images when close the script
                cv2.destroyAllWindows()
                break

    # Merge data of two numpy files
    def merge_data(self, file1, file2):
        data_file1 = list(np.load('Data/' + file1))
        data_file2 = list(np.load('Data/' + file2))

        total_data = data_file1 + data_file2

        np.save(self.path_file,total_data)

        print("Number of examples: ", len(total_data))

    def save_data(self):
        np.save(self.path_file,self.training_data)
        print('File saved!')

	# Collecting training data
    def CollectingTrainingData(self):
        keys = Keys()
        screen = Screen()
        agent = Agent()
        print('Starting Training in...')

		# Countdown to start the training
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)
        
        paused = False
        last_time = time.time()
		
        while True:
	
            if not paused:
                grabbed_screen = screen.GrabScreen() # Get actual frame
                keys_pressed = keys.KeyCheck() # Check for pressed keys 
                output_action = keys.KeysActionFreeKicksOutput(keys_pressed) # Verifies if one action key was pressed
                if output_action != [0,0,0,0]:
                    self.training_data.append([grabbed_screen,output_action]) # Create an instance of training data
				
            if output_action == [1,0,0,0]:
                print('Left')
                agent.left()
            elif output_action == [0,1,0,0]:
                print('Right')
                agent.right()
            elif output_action == [0,0,1,0]:
                print('Low Kick')
                agent.low_shoot()
            elif output_action == [0,0,0,1]:
                print('High Kick')
                agent.high_shoot()
            
            if len(self.training_data) % 100 == 0 and len(self.training_data) > 0:
                print(len(self.training_data))
                
            keys_pressed = keys.KeyCheck()
			
			# Pausing or Unpausing training
            if 'Q' in keys_pressed:
                if paused:
                    paused = False
                    print('Unpausing training...')
                    time.sleep(2)
                else:
                    print('Pausing training!')
                    paused = True
                    time.sleep(1)
			
			# Saving Data
            if 'R' in keys_pressed:
                np.save(self.path_file,self.training_data)

    # Visualize training data
    def VisualizeTrainingData(self):
        lefts = []
        rights = []
        low_shoots = []
        high_shoots = []

        print(np.array(self.training_data[1])[0].shape)

        for data in self.training_data:
            img = data[0]
            action = data[1]

            #print(img, ' - ', action)
            #cv2.imwrite('test.png', img[40:850,0:1400])

            ## Complete Movement examples
            if action == [1,0,0,0]:
                lefts.append([img,action])
            elif action == [0,1,0,0]:
                rights.append([img,action])
            elif action == [0,0,1,0]:
                low_shoots.append([img,action])
            elif action == [0,0,0,1]:
                high_shoots.append([img,action])
            else:
                print('No matches corresponding to a legal action')

        print('New data length: ', len(self.training_data))
        print('Lefts: ', len(lefts))
        print('Rights: ', len(rights))
        print('Low Shoots: ', len(low_shoots))
        print('High Shoots: ', len(high_shoots))

    # Create training data
    def CreateTrainingData(self):
        color_data = []
        grayscale_data = []
        object_detection_data = []

        cnn_graph = CNN_V2_With_Barrier()

        for data in self.training_data:
            img = data[0][40:850,0:1400]
            action = data[1]

            ## Grayscale
            new_screen_grayscaled = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (120,90)) # Converting and Resizing frame
            normalized_grayscaled_screen = new_screen_grayscaled.astype('float32') / 255 # Normalizing
            grayscale_data.append([normalized_grayscaled_screen,action])


            ## RGB
            new_screen_rgb = cv2.resize(img, (120,90)) # Converting and Resizing frame
            normalized_rgb_screen = new_screen_rgb.astype('float32') / 255 # Normalizing
            color_data.append([normalized_rgb_screen, action])


            ## Object Detection (Without Goalkeeper)
            new_screen_od = cnn_graph.get_image_feature_map(img) # Extract a 128 feature map
            object_detection_data.append([new_screen_od, action])


        print('Color data length: ', len(color_data))
        print('Grayscale data length: ', len(grayscale_data))
        print('Object Detection data length: ', len(object_detection_data))

        # Saving new data
        np.save('Data/fifa_fk_grayscale_data.npy', grayscale_data)
        np.save('Data/fifa_fk_color_data.npy', color_data)
        np.save('Data/fifa_fk_object_detection_data.npy', object_detection_data)
    
    # Create training data
    def CreateTrainingDataWithGK(self):
        color_data = []
        grayscale_data = []
        object_detection_data = []
        object_detection_only_gk_data = []
        object_detection_with_gk_all_data = []

        cnn_gk_graph = CNN_Only_Goalkeeper()
        cnn_gk_all_graph = CNN_Fk_With_Goalkeeper_All()

        for data in self.training_data:
            img = data[0][40:850,0:1400]
            action = data[1]

            ## Grayscale
            new_screen_grayscaled = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (120,90)) # Converting and Resizing frame
            normalized_grayscaled_screen = new_screen_grayscaled.astype('float32') / 255 # Normalizing
            grayscale_data.append([normalized_grayscaled_screen,action])


            ## RGB
            new_screen_rgb = cv2.resize(img, (120,90)) # Converting and Resizing frame
            normalized_rgb_screen = new_screen_rgb.astype('float32') / 255 # Normalizing
            color_data.append([normalized_rgb_screen, action])


            ## Object Detection (Only Goalkeeper)
            #new_screen_od_only_gk = cnn_gk_graph.get_image_feature_map(img) # Extract a 128 feature map
            #object_detection_only_gk_data.append([new_screen_od_only_gk, action])

            ## Object Detection (Goalkeeper and All elements)
            new_screen_od_with_gk_all = cnn_gk_all_graph.get_image_feature_map(img) # Extract a 128 feature map
            object_detection_with_gk_all_data.append([new_screen_od_with_gk_all, action])


        print('Color data length: ', len(color_data))
        print('Grayscale data length: ', len(grayscale_data))
        print('Object Detection Only GK data length: ', len(object_detection_only_gk_data))

        # Saving new data
        #np.save('Data/fifa_fk_with_gk_grayscale_data.npy', grayscale_data)
        #np.save('Data/fifa_fk_with_gk_color_data.npy', color_data)
        #np.save('Data/fifa_fk_with_gk_object_detection_only_gk_data.npy', object_detection_only_gk_data)
        np.save('Data/fifa_fk_with_gk_object_detection_with_gk_all_data.npy', object_detection_with_gk_all_data)

if __name__ == '__main__':
    data = Data('training_data_with_gk.npy')
    #data.CollectingTrainingData()
    #data.VisualizeTrainingData()
    #data.CreateTrainingData()
    data.CreateTrainingDataWithGK()
    #data.validate_data()
    #data.BalanceData()
    #data.merge_data('balanced_training_data_freekicks_with_goalkeeper_od_1ksamples.npy', 'balanced_training_data_freekicks_with_goalkeeper_od_2.npy')
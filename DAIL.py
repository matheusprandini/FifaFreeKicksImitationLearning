import cv2
import numpy as np
import scipy.stats
import time
from Data import Data
from math import log, e
from Keys import Keys
from Screen import Screen
from FifaEnv import FifaEnv
from CNN_Chintan_V2_With_Barrier import CNN_V2_With_Barrier
from CNN_Fk_With_Goalkeeper_All import CNN_Fk_With_Goalkeeper_All
import winsound

# Deep Active Imitation Learning Class
class DAIL():

    ## Behavioral Agent: agent trained with behavioral cloning on the specified Dataset
    ## Active Sample Size: size (in percentage) of active samples of training data
    ## Dataset Behavioral Cloning: data used to train the behavioral agent
    def __init__(self, behavioral_agent, active_sample_size, threshold, dataset_behavioral_cloning, new_data_name):
        self.behavioral_agent = behavioral_agent
        self.active_sample_size = active_sample_size
        self.threshold = threshold
        self.dataset_behavioral_cloning = dataset_behavioral_cloning
        self.active_samples = Data(new_data_name)

    def query_sound(self):
        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)

    def calculate_entropy(self, labels):
        """ Computes entropy of label distribution. """

        entropy = scipy.stats.entropy(labels,base=2)  # input probabilities to get the entropy 
        return entropy

    def preprocess_image(self, image, grayscale_mode):

        image = image[40:850,0:1400]

        if grayscale_mode:
            new_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (120,90)) # Converting and Resizing frame # Converting frame
            preprocessed_image = new_image.astype('float32') / 255 # Normalizing
            final_image = preprocessed_image.reshape((-1,90,120,1)) # Resizing frame
        else:
            new_image = cv2.resize(image, (120,90)) # Converting and Resizing frame
            preprocessed_image = new_image.astype('float32') / 255 # Normalizing
            final_image = preprocessed_image.reshape((-1,90,120,3)) # Resizing frame

        return final_image, preprocessed_image, image

    def create_active_samples(self, active_samples, grayscale_mode, object_detection, without_gk):
        color_data = []
        grayscale_data = []
        od_data = []

        if object_detection:
            if without_gk:
                cnn_graph = CNN_V2_With_Barrier()
            else:
                cnn_graph = CNN_Fk_With_Goalkeeper_All()

        for data in active_samples:
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

            ## Object Detection
            if object_detection:
                new_screen_od, _ = cnn_graph.get_image_feature_map(img) # Extract a 128 feature map
                od_data.append([new_screen_od, action])

        print('Color data length: ', len(color_data))
        print('Grayscale data length: ', len(grayscale_data))
        print('OD data length: ', len(od_data))

        if object_detection:
                return od_data
        elif grayscale_mode:
            return grayscale_data
        else:
            return color_data

    def execute_active_deep_imitation_learning_free_kicks(self, grayscale_mode=True, object_detection=False, without_gk=True):
        
        game_env = FifaEnv()
        keys = Keys()
        screen = Screen()

        if object_detection:
            if without_gk:
                cnn_graph = CNN_V2_With_Barrier()
            else:
                cnn_graph = CNN_Fk_With_Goalkeeper_All()

        print('Starting Active Deep Imitation Learning in...')

        # Countdown to start running the agent  
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)
        
        paused = False

        number_active_samples_to_reach = self.active_sample_size * len(self.dataset_behavioral_cloning.training_data) / 100.0

        print(len(self.active_samples.training_data))
        print(number_active_samples_to_reach)

        cont = 0

        while len(self.active_samples.training_data) < number_active_samples_to_reach:
            
            if not paused:
                # Verifies if it's an end of the training session (Time is over) or if there's a bug
                end_training_session = game_env.check_end_of_episode()
                bug = game_env.check_bug()
                
                if end_training_session or bug:
                    game_env.hard_reset()
                
                while bug:
                    bug = game_env.check_bug()

                game_over = False

                time.sleep(3)
                
                # get the current state
                x_t = game_env.observe_state()

                if object_detection:
                    s_t, image = cnn_graph.get_image_feature_map(x_t)
                    s_t = s_t.astype('float32') / 6
                else:
                    s_t, preprocessed_image, image = self.preprocess_image(x_t, grayscale_mode)
                    s_t = np.array(s_t)

                while not game_over:

                    #cv2.imwrite(str(cont)+'.png',image)
                    #cont+=1
                    
                    #### Get next action ####		

                    # Prediction and Entropy of the probabilities
                    action_predicted, action_probabilities = self.behavioral_agent.predict_action(s_t)
                    entropy = self.calculate_entropy(action_probabilities)
                    print(action_probabilities, ' -> ', entropy)

                    if entropy < self.threshold:
                        ## Execute the behavioral agent's action
                        x_t, r_t, game_over = game_env.step(action_predicted)

                    else:
                        ## Queries a non-expert action
                        self.query_sound()

                        print('Query non-expert action: ')
                        self.behavioral_agent.release_actions()

                        # Waiting for a non-expert action
                        while True:
                            keys_pressed = keys.KeyCheck() # Check for pressed keys
                            non_expert_action = keys.KeysActionFreeKicksOutput(keys_pressed) # Verifies if one move key was pressed
                            if non_expert_action != [0, 0, 0, 0]:
                                print(non_expert_action)
                                self.active_samples.training_data.append([image, non_expert_action])

                                action = np.argmax(non_expert_action)
                                x_t, r_t, game_over = game_env.step(action)
                                break

                    if object_detection:
                        s_t, image = cnn_graph.get_image_feature_map(x_t)
                        s_t = s_t.astype('float32') / 6
                    else:
                        s_t, preprocessed_image, image = self.preprocess_image(x_t, grayscale_mode)
                        s_t = np.array(s_t)
                        
                time.sleep(2)
            
            keys_pressed = keys.KeyCheck()
            
            if 'Q' in keys_pressed:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    self.behavioral_agent.release_moves()
                    paused = True
                    time.sleep(1)
        
        ## Aggregating data and saving
        self.active_samples.training_data = self.dataset_behavioral_cloning.training_data + self.create_active_samples(self.active_samples.training_data, grayscale_mode, object_detection, without_gk)
        self.active_samples.save_data()
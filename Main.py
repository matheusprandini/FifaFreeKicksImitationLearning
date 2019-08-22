import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
from FifaEnv import FifaEnv
from Agent import Agent
from Data import Data
from DAIL import DAIL
from CNN_Chintan_V2_With_Barrier import CNN_V2_With_Barrier
from CNN_Fk_With_Goalkeeper_All import CNN_Fk_With_Goalkeeper_All

# Initialize Global Parameters
DATA_DIR = "Models/"
NUM_EPOCHS_TEST = 100

def print_action(action):

    if action == 0:
        print('Left')
    elif action == 1:
        print('Right')
    elif action == 2:
        print('Low Kick')
    else:
        print('High Kick')
    
def preprocess_image(image, grayscale_mode=True):
    if grayscale_mode:
        new_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (120,90)) # Converting and Resizing frame # Converting frame
        preprocessed_image = new_image.astype('float32') / 255 # Normalizing
        final_image = preprocessed_image.reshape((-1,90,120,1)) # Resizing frame
    else:
        new_image = cv2.resize(image, (120,90)) # Converting and Resizing frame
        preprocessed_image = new_image.astype('float32') / 255 # Normalizing
        final_image = preprocessed_image.reshape((-1,90,120,3)) # Resizing frame

    return np.array(final_image), image


##################### TEST DIL AGENTS #####################

## WITHOUT GK

def test_direct_imitation_agent_free_kicks(grayscale_mode=True, object_detection=False, debug=True):

    game_env = FifaEnv()
    direct_imitation_agent = Agent()
    cnn_graph = CNN_V2_With_Barrier()

    ## Loading model (grayscale or color or object detection)
    if grayscale_mode:
        direct_imitation_agent.load_action_model("Models/fifa_fk_grayscale_dil_agent_v2.h5")
    elif object_detection:
        #direct_imitation_agent.load_action_model("Models/fifa_fk_object_detection_dil_agent_v2.h5")
        direct_imitation_agent.load_action_model("Models/fifa_fk_object_detection_agent.h5")
    else:
        direct_imitation_agent.load_action_model("Models/fifa_fk_color_dil_agent_v2.h5")
	
    num_goals = 0
    cont = 0
	
    print('----- TESTING DIRECT IMITATION AGENT -----')

    for e in range(NUM_EPOCHS_TEST):

        game_over = False
        goal = 0

        time.sleep(1.5)
		
        # Verifies if it's an end of the training session (Time is over) or if there's a bug
	
        end_training_session = game_env.check_end_of_episode()
        bug = game_env.check_bug()
		
        if end_training_session or bug:
            game_env.hard_reset()
		
        while bug:
            bug = game_env.check_bug()
		
        # get first state
        x_t = game_env.observe_state()[40:850,0:1400]

        if object_detection:
            s_t, _ = cnn_graph.get_image_feature_map(x_t)
            s_t = s_t.astype('float32') / 6
        else:
            s_t, image = preprocess_image(x_t, grayscale_mode)
	
        while not game_over:

            #cv2.imwrite(str(cont)+'.png',image)
            #cont+=1
			
            #### Get next action ####		

            # Best action
            a_t, _ = direct_imitation_agent.predict_action(s_t)
            if debug:
                print_action(a_t)

            # apply action, get reward
            x_t, r_t, game_over = game_env.step(a_t)

            if object_detection:
                s_t, _ = cnn_graph.get_image_feature_map(x_t)
                s_t = s_t.astype('float32') / 6
            else:
                s_t, image = preprocess_image(x_t, grayscale_mode)
			
		    # increment goal if it's a goal
            if r_t == 1:
                goal += 1
				
        time.sleep(2)
			
        num_goals += goal
        
    print("Epoch {:04d}/{:d} | Total Goals: {:d}"
        .format(e + 1, NUM_EPOCHS_TEST, num_goals))

    return num_goals


## WITH GK

def test_direct_imitation_agent_free_kicks_with_gk(grayscale_mode=True, object_detection=False, debug=True):

    game_env = FifaEnv()
    direct_imitation_agent = Agent()
    cnn_graph = CNN_Fk_With_Goalkeeper_All()

    ## Loading model (grayscale or color or object detection)
    if grayscale_mode:
        direct_imitation_agent.load_action_model("Models/fifa_fk_with_gk_grayscale_dil_agent.h5")
    elif object_detection:
        direct_imitation_agent.load_action_model("Models/fifa_fk_with_gk_object_detection_agent.h5")
    else:
        direct_imitation_agent.load_action_model("Models/fifa_fk_with_gk_color_dil_agent.h5")
	
    num_goals = 0
    cont = 0
	
    print('----- TESTING DIRECT IMITATION AGENT -----')

    for e in range(NUM_EPOCHS_TEST):

        game_over = False
        goal = 0

        time.sleep(1.5)
		
        # Verifies if it's an end of the training session (Time is over) or if there's a bug
	
        end_training_session = game_env.check_end_of_episode()
        bug = game_env.check_bug()
		
        if end_training_session or bug:
            game_env.hard_reset()
		
        while bug:
            bug = game_env.check_bug()
		
        # get first state
        x_t = game_env.observe_state()[40:850,0:1400]

        if object_detection:
            s_t, _ = cnn_graph.get_image_feature_map(x_t)
            s_t = s_t.astype('float32') / 6
        else:
            s_t, image = preprocess_image(x_t, grayscale_mode)
	
        while not game_over:

            #cv2.imwrite(str(cont)+'.png',image)
            #cont+=1
			
            #### Get next action ####		

            # Best action
            a_t, _ = direct_imitation_agent.predict_action(s_t)
            if debug:
                print_action(a_t)

            # apply action, get reward
            x_t, r_t, game_over = game_env.step(a_t)

            if object_detection:
                s_t, _ = cnn_graph.get_image_feature_map(x_t)
                s_t = s_t.astype('float32') / 6
            else:
                s_t, image = preprocess_image(x_t, grayscale_mode)
			
		    # increment goal if it's a goal
            if r_t == 1:
                goal += 1
				
        time.sleep(2)
			
        num_goals += goal
        
    print("Epoch {:04d}/{:d} | Total Goals: {:d}"
        .format(e + 1, NUM_EPOCHS_TEST, num_goals))

    return num_goals


##################### TEST DAI AGENTS #####################

## WITHOUT GK

def test_dai_agent_free_kicks(grayscale_mode=True, object_detection=False, debug=True):

    game_env = FifaEnv()
    deep_active_imitation_agent = Agent()
    cnn_graph = CNN_V2_With_Barrier()

    ## Loading model (grayscale or color or object detection)
    if grayscale_mode:
        deep_active_imitation_agent.load_action_model("Models/fifa_fk_grayscale_dai_agent_v2.h5")
    elif object_detection:
        #deep_active_imitation_agent.load_action_model("Models/fifa_fk_object_detection_dai_agent_v2.h5")
        deep_active_imitation_agent.load_action_model("Models/fifa_fk_object_detection_dai_agent.h5")
    else:
        deep_active_imitation_agent.load_action_model("Models/fifa_fk_color_dai_agent_v2.h5")
	
    num_goals = 0
    cont = 0
	
    print('----- TESTING DEEP ACTIVE IMITATION AGENT -----')

    for e in range(NUM_EPOCHS_TEST):

        game_over = False
        goal = 0

        time.sleep(1.5)
		
        # Verifies if it's an end of the training session (Time is over) or if there's a bug
	
        end_training_session = game_env.check_end_of_episode()
        bug = game_env.check_bug()
		
        if end_training_session or bug:
            game_env.hard_reset()
		
        while bug:
            bug = game_env.check_bug()
		
        # get first state
        x_t = game_env.observe_state()[40:850,0:1400]
        
        if object_detection:
            s_t, _ = cnn_graph.get_image_feature_map(x_t)
            s_t = s_t.astype('float32') / 6
        else:
            s_t, image = preprocess_image(x_t, grayscale_mode)
	
        while not game_over:

            #cv2.imwrite(str(cont)+'.png',image)
            #cont+=1
			
            #### Get next action ####		

            # Best action
            a_t, _ = deep_active_imitation_agent.predict_action(s_t)
            if debug:
                print_action(a_t)

            # apply action, get reward
            x_t, r_t, game_over = game_env.step(a_t)

            if object_detection:
                s_t, _ = cnn_graph.get_image_feature_map(x_t)
                s_t = s_t.astype('float32') / 6
            else:
                s_t, image = preprocess_image(x_t, grayscale_mode)
			
		    # increment goal if it's a goal
            if r_t == 1:
                goal += 1
				
        time.sleep(2)
			
        num_goals += goal
        
    print("Epoch {:04d}/{:d} | Total Goals: {:d}"
        .format(e + 1, NUM_EPOCHS_TEST, num_goals))

    return num_goals


## WITH GK

def test_dai_agent_free_kicks_with_gk(grayscale_mode=True, object_detection=False, debug=True):

    game_env = FifaEnv()
    deep_active_imitation_agent = Agent()
    cnn_graph = CNN_Fk_With_Goalkeeper_All()

    ## Loading model (grayscale or color or object detection)
    if grayscale_mode:
        deep_active_imitation_agent.load_action_model("Models/fifa_fk_with_gk_grayscale_dai_agent.h5")
    elif object_detection:
        deep_active_imitation_agent.load_action_model("Models/fifa_fk_with_gk_object_detection_dai_agent.h5")
    else:
        deep_active_imitation_agent.load_action_model("Models/fifa_fk_with_gk_color_dai_agent.h5")
	
    num_goals = 0
    cont = 0
	
    print('----- TESTING DEEP ACTIVE IMITATION AGENT -----')

    for e in range(NUM_EPOCHS_TEST):

        game_over = False
        goal = 0

        time.sleep(1.5)
		
        # Verifies if it's an end of the training session (Time is over) or if there's a bug
	
        end_training_session = game_env.check_end_of_episode()
        bug = game_env.check_bug()
		
        if end_training_session or bug:
            game_env.hard_reset()
		
        while bug:
            bug = game_env.check_bug()
		
        # get first state
        x_t = game_env.observe_state()[40:850,0:1400]
        
        if object_detection:
            s_t, _ = cnn_graph.get_image_feature_map(x_t)
            s_t = s_t.astype('float32') / 6
        else:
            s_t, image = preprocess_image(x_t, grayscale_mode)
	
        while not game_over:

            cv2.imwrite(str(cont)+'.png', image)
            cont+=1
			
            #### Get next action ####		

            # Best action
            a_t, _ = deep_active_imitation_agent.predict_action(s_t)
            if debug:
                print_action(a_t)

            # apply action, get reward
            x_t, r_t, game_over = game_env.step(a_t)

            if object_detection:
                s_t, _ = cnn_graph.get_image_feature_map(x_t)
                s_t = s_t.astype('float32') / 6
            else:
                s_t, image = preprocess_image(x_t, grayscale_mode)
			
		    # increment goal if it's a goal
            if r_t == 1:
                goal += 1
				
        time.sleep(2)
			
        num_goals += goal
        
    print("Epoch {:04d}/{:d} | Total Goals: {:d}"
        .format(e + 1, NUM_EPOCHS_TEST, num_goals))

    return num_goals


##################### TRAINING DAI AGENTS #####################

## WITHOUT GK

def training_fk_color_dai_agent():
    agent = Agent()
    agent.load_action_model("Models/fifa_fk_color_dil_agent_v2.h5")
	
    data = Data('fifa_fk_color_data_v2.npy')

    dail = DAIL(agent, 5, 1.5, data, "new_training_data_5%_fifa_fk_color_data_v2.npy")
    dail.execute_active_deep_imitation_learning_free_kicks(False)

def training_fk_grayscale_dai_agent():
    agent = Agent()
    agent.load_action_model("Models/fifa_fk_grayscale_dil_agent_v2.h5")
	
    data = Data('fifa_fk_grayscale_data_v2.npy')

    dail = DAIL(agent, 5, 1.5, data, "new_training_data_5%_fifa_fk_grayscale_data_v2.npy")
    dail.execute_active_deep_imitation_learning_free_kicks()

def training_fk_object_detection_dai_agent():
    agent = Agent()
    agent.load_action_model("Models/fifa_fk_object_detection_agent.h5")
	
    data = Data('fifa_fk_object_detection_data_v2.npy')

    dail = DAIL(agent, 5, 1.5, data, "new_training_data_5%_fifa_fk_object_detection_data.npy")
    dail.execute_active_deep_imitation_learning_free_kicks(False, True)


## WITH GK

def training_fk_with_gk_color_dai_agent():
    agent = Agent()
    agent.load_action_model("Models/fifa_fk_with_gk_color_dil_agent.h5")
	
    data = Data('fifa_fk_with_gk_color_data.npy')

    dail = DAIL(agent, 5, 1.5, data, "new_training_data_5%_fifa_fk_with_gk_color_data.npy")
    dail.execute_active_deep_imitation_learning_free_kicks(False, False, False)

def training_fk_with_gk_grayscale_dai_agent():
    agent = Agent()
    agent.load_action_model("Models/fifa_fk_with_gk_grayscale_dil_agent.h5")
	
    data = Data('fifa_fk_with_gk_grayscale_data.npy')

    dail = DAIL(agent, 5, 1.6, data, "new_training_data_5%_fifa_fk_with_gk_grayscale_data.npy")
    dail.execute_active_deep_imitation_learning_free_kicks(True, False, False)

def training_fk_with_gk_object_detection_dai_agent():
    agent = Agent()
    agent.load_action_model("Models/fifa_fk_with_gk_object_detection_agent.h5")
	
    data = Data('fifa_fk_with_gk_object_detection_with_gk_all_data.npy')

    dail = DAIL(agent, 5, 1.3, data, "new_training_data_5%_fifa_fk_with_gk_object_detection_data.npy")
    dail.execute_active_deep_imitation_learning_free_kicks(False, True, False)

#for i in range(10):
    #test_dai_agent_free_kicks(True, False)

#training_fk_color_dail_agent()

#training_fk_grayscale_dai_agent()
#test_dai_agent_free_kicks()

#test_direct_imitation_agent_free_kicks(False,True,True)

#training_fk_object_detection_dai_agent()

'''scores = []

for i in range(10):
    goals = test_direct_imitation_agent_free_kicks_with_gk(False,True,False)
    scores.append(goals)

print(scores)'''

#training_fk_with_gk_color_dai_agent()
#training_fk_with_gk_grayscale_dai_agent()
#training_fk_with_gk_object_detection_dai_agent()

'''scores = []

for i in range(10):
    goals = test_dai_agent_free_kicks_with_gk(False,True,False)
    scores.append(goals)

print(scores)'''

test_dai_agent_free_kicks_with_gk(False,False,False)
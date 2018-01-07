from agent_dir.agent import Agent
import scipy
import numpy as np
# import os
# import gym
# from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Input, Dense, Reshape
# from keras.optimizers import RMSprop
# from keras.layers.core import Activation, Dropout, Flatten
# from keras.layers.convolutional import UpSampling2D, Convolution2D


#Script Parameters
input_dim = 80 * 80
gamma = 0.99
update_frequency = 1
learning_rate = 1e-4
# resume = False
render = False


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py

    Input:
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array
        Grayscale image, shape: (80, 80, 1)

    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


def pong_preprocess_screen(I):
    I = I[35:195] 
    I = I[::2,::2,0] 
    I[I == 144] = 0 
    I[I == 109] = 0 
    I[I != 0] = 1 
    return I.astype(np.float).ravel()

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#Define the main model (WIP)
def learning_model(input_dim=80*80):
    model = Sequential()
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding='same', activation='relu'))
    model.add(Convolution2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    opt = RMSprop(lr=learning_rate, decay=0.99)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
#     if resume == True:
#         model.load_weights('pong_model_checkpoint.h5')
    return model


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            print('loading trained model')
#             import keras
#             self.model = keras.models.load_model('model.h5')

        ##################
        # YOUR CODE HERE #
        ##################
        
        self.env = env

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

#         self.prev_x = None


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        # GPU usage
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=config))

        #Initialize
#         number_of_inputs = self.env.action_space.n #This is incorrect for Pong (?)
        observation = self.env.reset()
        prev_x = None
        xs, dlogps, drs, probs = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        train_X = []
        train_y = []
        
        model = learning_model()

        #Begin training
        while True:
            if render: 
                self.env.render()
            #Preprocess, consider the frame difference as features
            cur_x = pong_preprocess_screen(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
            prev_x = cur_x
            #Predict probabilities from the Keras model
            aprob = ((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
            #aprob = aprob/np.sum(aprob)
            #Sample action
            #action = np.random.choice(6, 1, p=aprob)
            #Append features and labels for the episode-batch
            xs.append(x)
            probs.append((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
            aprob = aprob/np.sum(aprob)
            action = np.random.choice(6, 1, p=aprob)[0]
            y = np.zeros([6])
            y[action] = 1
            #print action
            dlogps.append(np.array(y).astype('float32') - aprob)
            observation, reward, done, info = self.env.step(action)
            reward_sum += reward
            drs.append(reward) 
#             print(reward, end=' ')
            if done:
                episode_number += 1
                epx = np.vstack(xs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)
                discounted_epr = discount_rewards(epr)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                epdlogp *= discounted_epr
                #Slowly prepare the training batch
                train_X.append(xs) 
                train_y.append(epdlogp)
                xs,dlogps,drs = [],[],[]
                #Periodically update the model
                if episode_number % update_frequency == 0: 
                    y_train = probs + learning_rate * np.squeeze(np.vstack(train_y)) #Hacky WIP
                    #y_train[y_train<0] = 0
                    #y_train[y_train>1] = 1
                    #y_train = y_train / np.sum(np.abs(y_train), axis=1, keepdims=True)
        #             print('Training Snapshot:')
        #             print(y_train)
                    model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
                    #Clear the batch
                    train_X = []
                    train_y = []
                    probs = []
                    #Save a checkpoint of the model
        #             os.remove('pong_model_checkpoint.h5') if os.path.exists('pong_model_checkpoint.h5') else None
        #             model.save_weights('pong_model_checkpoint.h5')
                #Reset the current environment nad print the current results
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print()
                print('Environment reset imminent. Total Episode Reward: %f. Running Mean: %f' % (reward_sum, running_reward))
                model.save('model.h5')
                reward_sum = 0
                observation = self.env.reset()
                prev_x = None
            if reward != 0:
                print('Episode %d Result: ' % episode_number, 'Defeat!' if reward == -1 else 'VICTORY!')


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()
    
#         cur_x = pong_preprocess_screen(observation)
#         x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(input_dim)
#         self.prev_x = cur_x
#         #Predict probabilities from the Keras model
#         action = self.model.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten().argmax()
#         return action

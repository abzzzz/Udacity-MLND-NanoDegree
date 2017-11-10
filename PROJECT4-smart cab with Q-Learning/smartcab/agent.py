# -*- coding:utf-8 -*-

import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.count = 0


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update additional class parameters as needed
        self.count = self.count + 1 #计数器 实验数目

        if testing==True:# If 'testing' is True, set epsilon and alpha to 0

            self.epsilon=0
            self.alpha=0

        else:# Update epsilon using a decay function of your choice
            #self.epsilon = self.epsilon - 0.05 #按题目来的第一次
            self.epsilon = math.pow(0.99,self.count) # 次数方越大越好
            #self.epsilon = math.cos(0.005 * self.count) # cos 中的系数越小越好

            self.alpha=0.18
            #我首先试了0.6效果很不好，然后又尝试0.13 效果变成了CB 然后微调为0.18 效果A A+

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint,inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        

        maxQ = None

        maxQ = max(self.Q[state].values()) # Calculate the maximum Q-value of all actions for a given state

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########


        # When learning, check if the 'state' is not in the Q-table
        if self.learning == True:
            # If it is not
            if state not in self.Q:
                #create a new dictionary for that state
                self.Q[state] = dict()
                
                #Then, for each action available
                for action in self.valid_actions:
                    #set the initial Q-value to 0.0
                    self.Q[state][action] = 0.0

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None
        maxmax = []

        ########### 
        ## TO DO ##
        ###########
        

        if self.learning == False:# When not learning

            random_action = random.randint(0, 4)#有四个动作可以选择[None, 'forward', 'left', 'right']

            action = self.env.valid_actions[random_action] #choose a random action
        
        else:# When learning
            if self.epsilon>random.random():#choose a random action with 'epsilon' probability
                action = random.choice(self.valid_actions)
            
            #Otherwise, choose an action with the highest Q-value for the current state
            else:
                max_Q = self.get_maxQ(state)#首先是读取最大的Q值

                aQ = self.Q[state]#然后是读取动作

                for i in self.valid_actions:
                    #在执行的动作中循环
                    if aQ[i] == max_Q:#如果找到有Q等于最大的Q 即相当于目前最好的策略
                        maxmax.append(i)#就保存到一个列表中去，因为可能会有多个不同策略但是Q的值很高
                
                action = random.choice(maxmax) #为了避免人为的选择对于实验造成影响，在有多个最优的状态动作时随机进行选择


        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        #implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        if self.learning==True: #When learning

            self.Q[state][action] = ( 1 - self.alpha ) * self.Q[state][action] + self.alpha * reward
            #self.Q[state][action] = self.Q[state][action] + self.alpha * (reward - self.Q[state][action])

        else:

            pass

        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning = True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.01, log_metrics = True, display = False, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = 20, tolerance = 0.005)


if __name__ == '__main__':
    run()
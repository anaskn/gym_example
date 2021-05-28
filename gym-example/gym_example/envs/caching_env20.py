import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
from ray.rllib.env.env_context import EnvContext



class Caching_v020(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""
   

    def __init__(self, config: EnvContext):
        self.action_space = Box(low= 0, high= 1 ,shape=(20,), dtype=np.float32)
        self.observation_space = Box(low= 0, high= 100, shape=(20,3), dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(0)

        self.ttl_var = config["ttl_var"]
        self.variable = config["variable"]

        self.neighbor = config["nei_tab"]
        self.request = config["lst_tab"]

        self.reward_cumul = []

        lst = self.request

        tab_cache= []
        tab_request = []
        nei_req = []
        cache_on_tab = []
        neighbor_number_tab = []
        ttl_tab = []
        for xx in range(20):
            tab_cache.append(50) 
            tab_request.append(lst[xx])
            nei_req.append(-99)
            cache_on_tab.append(0)
            neighbor_number_tab.append(0)
            ttl_tab.append(np.zeros(20))
        
        self.caching_cap =  tab_cache 
        self.request = tab_request
        self.neigbors_request = nei_req
        self.cache_on = cache_on_tab
        self.neighbor_number = neighbor_number_tab
        self.ttl = ttl_tab

        self.unused_shared=None 
        self.unused_own = None
        self.unsatisfied_shared= None
        self.unsatisfied_own= None

        self.epochs_num=0
        self.steps = 0

    def next_obs(self,i):
        
        #if i == 3:
            #self.steps = self.steps+1
            #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx == ", self.steps)
        nei_tab = self.neighbor#ret_nei(self.cpt)# args must be variable
        ttl_var = self.ttl_var # must be variable
        
        entity_pos = []

        for x in range(len(self.caching_cap)):
            lstt= []

            lstt.append(self.request[x][i])

            #init  caching_cap
            if i == 0 :
                self.caching_cap[x]=50
                lstt.append(50)
            else:           
                if i-ttl_var > 0:
                    self.caching_cap[x] = self.caching_cap[x] + self.ttl[x][i-ttl_var]
                    
                min_val = min(self.caching_cap[x] , float(self.cache_on[x]))
                self.caching_cap[x] = self.caching_cap[x] - min_val  

                self.ttl[x][i] = min_val
                lstt.append(self.caching_cap[x])
            
            #init  neigbors_request
            cache = 0
            for y in range(len(nei_tab[i][x])):

                if len(nei_tab[i][y]) == 0:
                    cache = cache + 0
                
                else:
                    cache = cache + (self.request[nei_tab[i][x][y]][i]/len(nei_tab[i][nei_tab[i][x][y]]) )

            if len(nei_tab[i][x])==0:
                self.neigbors_request[x]= 0
                self.neighbor_number[x] = 0
                lstt.append(0)
            else:
                self.neigbors_request[x] = cache/len(nei_tab[i][x])
                self.neighbor_number[x] = len(nei_tab[i][x])
                lstt.append(self.neigbors_request[x])  

            entity_pos.append(lstt)

        entity_pos = np.array(entity_pos)
        return entity_pos

    def reset(self):
        self.epochs_num=0
        entity_pos = self.next_obs(0)
        return entity_pos

    def step(self, action):

        nei_tab = self.neighbor#ret_nei(self.cpt)
        self.epochs_num= self.epochs_num+1
        i = self.epochs_num
        entity_pos = self.next_obs(self.epochs_num)
        variable = self.variable 
        reward=[]
        R_c = variable[0]
        C = variable[1]
        fact_k = variable[2] 

        unused_shared = []
        unused_own = []
        nei_request_tab = []
        unsatisfied_shared = []
        unsatisfied_own =[]

        for zz in range(len(action)):
            cache1 = 0
            for y in range(len(nei_tab[i][zz])):

                if len(nei_tab[i][y]) == 0:
                    cache1= cache1 + 0
                else :
                    cache1=cache1+(max(0,(self.request[nei_tab[i][zz][y]][i]-((1-action[nei_tab[i][zz][y]])*self.caching_cap[nei_tab[i][zz][y]]))/len(nei_tab[i][nei_tab[i][zz][y]])) )
   
            if len(nei_tab[i][zz]) == 0 :
                cache1 = 0
           
            
            f = R_c * max(0, (1-action[zz]) * self.caching_cap[zz] )  \
               - C* ( max(0,  (self.request[zz][i]-(action[zz]*self.caching_cap[zz]))) + max(0, ( cache1 - (1-action[zz])*self.caching_cap[zz])/fact_k)  ) \
                  - C* ( max(0, ((action[zz]*self.caching_cap[zz])-self.request[zz][i])/fact_k) + max (0, ((1-action[zz])*self.caching_cap[zz]) - cache1) )  
        
            unused_shared.append( float(max(0,(1-action[zz])*self.caching_cap[zz] - cache1  )))
            unused_own.append( float(max(0, (action[zz] * self.caching_cap[zz])-self.request[zz][i] )))
            unsatisfied_shared.append(float(max(0,cache1 - (1-action[zz])*self.caching_cap[zz])))
            unsatisfied_own.append(float(max(0,self.request[zz][i] - action[zz]*self.caching_cap[zz])))

            reward.append(f)
        #init  self.cache_on[x]
        for zz in range(len(action)):
            self.cache_on[zz] = min(self.request[zz][i], ((action[zz]*100) * self.caching_cap[zz]) / 100.0)  \
                + min(self.neigbors_request[zz], (((1-action[zz])*100) * self.caching_cap[zz]) / 100.0)
        
        if self.epochs_num==19:
            done = True
        else:
            done = False


        thisdict = {
              "unused_shared": np.mean(unused_shared),
              "unused_own": np.mean(unused_own),
              "unsatisfied_shared": np.mean(unsatisfied_shared),
              "unsatisfied_own": np.mean(unsatisfied_own)
            }
        
        return entity_pos,np.mean(reward), done, thisdict
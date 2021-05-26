#!/usr/bin/env python
# encoding: utf-8

import gym
import gym_example
import pickle


def ret_lst(cpt):
    string1 =  'data4/listfile_40_'+str(cpt)+'.data' #_evol'+ , _pos'+
    with open(string1, 'rb') as filehandle:
    # read the data as binary data stream
        lst = pickle.load(filehandle)
    return lst

def ret_nei(cpt):
    string2 = 'data4/nei_tab_pos_40_'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)
    return nei_tab




if __name__ == "__main__":

    config = {
        "env": "caching-v0",  # or "corridor" if registered above
        "env_config": {
            "ttl_var": 3,
            "variable": [8,8,8,4],
            "nei_tab": ret_nei(1),
            "lst_tab": ret_lst(1),
                   
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        #"num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),

        #"model": {
            #"custom_model": "my_model",
        #    "custom_model": "keras_q_model",
        #    "vf_share_layers": True,
        #},

        "model": {
            # By default, the MODEL_DEFAULTS dict above will be used.

            # Change individual keys in that dict by overriding them, e.g.
            "fcnet_hiddens": [128, 128, 128],
            "fcnet_activation": [ "relu"],
            "vf_share_layers": True,
        },

        "lr": [1e-2],  # try different lrs
        "num_workers": 2,  # parallelism
        "framework": "torch"# if args.torch else "tf",
    }
    print("mmmmmmmmmmmmmmmmmmmmmm = ", config["env_config"]["env"])
    gym.make("caching-v0", config=config)
 

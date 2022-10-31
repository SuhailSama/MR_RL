
Repo contains: 
- "main.py" - has the necessary functions to generate a trajectory (or upload from file) and learn from it + testing. 
- "MR_env.py" - a gym env for MR simulator 
- "MR_simulator.py" - MR simulator 
- "Read_data.py" - reads experimental data from pkl (sent by Max). This needs to be modified accordingly 
- "Learning_Module.py" - Gaussian Process learning module 
- "MR_ddpg.py" - RL control using MDDPG 
- ""MR_experiment.py" - code for passing actions to the experiemtn and returning outputs (X,Y,alphas,time,freq)
- Scripts from Max 
- ...


______________________________________________
Good tutorial for modeling

https://towardsdatascience.com/a-beginners-guide-to-simulating-dynamical-systems-with-python-a29bc27ad9b1


Gym environment inspired by the ShipAI tutorial 


https://towardsdatascience.com/openai-gym-from-scratch-619e39af121f

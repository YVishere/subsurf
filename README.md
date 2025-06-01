It is a mess but I will clean up after I get it working.  
Relevant files: model.py, scripts.py, train.py, agent.py and game_env.py

The goal was to learn how to make an application specific AI agent trained with reinforcement learning  
A lot of optimization was done with GitHub Copilot. In the draft written completely by me I averaged at < 1 fps severly bottlenecking my training.  
  
Model slows down when a lot of GPU memory is used. This happens when RAM is used a lot and I avoided this by sending an (86,86) frame to the model. However this makes score tracking unreliable.
# Human Level Control Through Deep Reinforcement Learning


Soft attention based model for the task of action recognition in videos. Based on research paper [1](<https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning>) 

<img src="https://github.com/dtransposed/Paper-Implementation/blob/master/human_level_control_through_deep_reinforcement_learning/images/movie.gif">

## Running the project

To run the project:

1. Edit parameters ```DQN_brain.py``` and ```train.py``` to your liking. The implementation supports games Breakout and Pong.

2. Run```train.py``` to train the model.

3. Once done, the model can be evaluated using ```test.py``` and the plot of the training history can be saved using ```plot results.py```. The sample training history for Pong looks like this:

   <img src="https://github.com/dtransposed/Paper-Implementation/blob/master/human_level_control_through_deep_reinforcement_learning/images/pyplot_multiple_y-axis.png">

   

   


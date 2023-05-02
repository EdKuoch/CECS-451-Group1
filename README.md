Python 3.9.6 was used.

Packages installed:
- gymnasium
- gymnasium[other]
- gymnasium[box2d]
- tensorflow
- tensorflow-metal (for mac gpu) # THIS IS BAD, cpu is faster for this
- np

sudo apt install python3-dev python3-pip

Referenced:
- https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN

if by episode 100 it has not reached 50 rewards, i terminate it early 

trial A normal, no frames stacked, one gray input, network A











trial R, same as Q but change gamma back to 95 and remember that reward *5, tried *5, overfitted instantly, doing 1.5 instead, overfitted, learning rate from .01 to .0001, woopsy, should have done that, overfitted, time to do frozen model



break while others letting the epsilon move them slowly

a deep learning model powerful enough to learn all on its own all the little techniques, ig mine wasn't enough






probably .999 if i can in the future tbh, but whatever



obs a:
it sems like a lot of these models are either overfitted or they seem to prefer the brake option or the do nothing option

X trial P, same as O but increase positive rewards *5 overfitted 

X trial Q, same as O but increase positive rewards by *1.5 only overfitted

X trial O, same as N but memory size 1000 and changed negative rewards *5 every frame, frame stack 5 for more context, overfitted

X trial B, network fast with 64 batch size, 3 frames stacked together, thier action space, sticky frame

X trial C, same as B but network fast with 4 batch size and no frozen models, good so far, episode 124, q value around 6 and still getting low single digit rewards

X trial D, same as C but, my action space, 3 frames stack together and 2 batch size and memory only 50 so most recent ones only, episode 174, q values around 6/7, bleh

X trial E, same as D but 1 batch size???, q values updating very slowly, TERMINATED,  bad rewards even late ron, only like q value of 2

X trial F, same as E but 8 batch size???, q values updating very slowly, also epoch 4 so it would get more training, PROMISING, NAH NO REWARDS at episode 89

X trial G, same as E but 8 batch size???, q values updating very slowly, also epoch 4 so it would get more training, epsilon decay to .999 instead of .9999 and min epsilon to .02 instead of .01, no sticky fmaes, see obs a, overfitted

X trial H, same as E but 2 batch size with epoch 8???, no sticky frames, epsilon decay to .999 instead of .9999 and min epsilon to .02 instead of .01, ALSO OVERFITTED, but it simply learned to go left and center with gas to get the most reward, but performed the best nontheless

X trial I, same as H but with memory 10000

X trial K, same as J but WITH FROZEN MODEL and updating target every 5 episodes from 3, see obs a, overfitted.

X trial L, same as J but with 64 epoch sizes batch size 2, see obs a, overfitted


X trial M, same as L but gamma .99 so that it updates faster, UPDATES SO FAST WOW, like it started late but it got q values of 44 while the others were at 10s, then i realize that all of these q values for eac of the samples wer ethe same, that means with different states, its giving the same q values, heavy overfitting, so not great, see obs a, overfitted

X trial J, same as I but with memory 10000, epoch 16, same as batch size 32 epoch 1, see obs a, overfitted

X trial N, 10 reward, follows trial H, 8 batch size, only start training once step 50, epoch 1, memory size 50, trash this, memory size too small

The best model that I made figured out going left and center with gas was typically the best and assigned a lot of Q values for that. 


I wonder if any model will learn going relaly fast and then braking at the corners as to not lose control

Plot or somehow track the reward and q values and what not to track model, that's really important, i just did a mess of console print outs

lots of repeating q values that was sampled, so its probably good to do some sort of priority sampling from the replay buffer in the future, one would be prioritizing the more recent ones

as long as the best reward was somewhat recent, i know the mmodel is doing okay

initially the model goes left, and the models do well, but then the track starts to turn right and then the model fails there

there's so many repeated states that give the same q values, having a larger batch size allows us to have a less biased grab of everything, the weights get updated to that closer and closer

The other guy just did more rewards towards states that did full gas, which i think is a bit meh, im okay with adjusting the environment, but adjusting the actions themselves seem kind of like cheating, but it's definitely one approach to keep in mind
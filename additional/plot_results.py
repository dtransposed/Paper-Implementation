import matplotlib.pyplot as plt
import pickle

training_data = pickle.load(open("training_data.data",'rb'))

training_data = training_data[:-2]

episodes = [data[0] for data in training_data]
mean_q_values = [data[1] for data in training_data]
mean_rewards= [data[2] for data in training_data]
epsilons = [data[3] for data in training_data]

fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlim(min(episodes), max(episodes))
host.set_ylim(min(mean_rewards), max(mean_rewards))
par1.set_ylim(min(mean_q_values), max(mean_q_values))
par2.set_ylim(min(epsilons), max(epsilons))

host.set_xlabel("Episode")
host.set_ylabel("Mean Reward")
par1.set_ylabel("Mean Q-Value")
par2.set_ylabel("Epsilon")

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.9)

p1, = host.plot(episodes, mean_rewards, color=color1,label="Mean Reward")
p2, = par1.plot(episodes, mean_q_values, color=color2, label="Mean Q-Value")
p3, = par2.plot(episodes, epsilons, color=color3, label="Epsilon")

lns = [p1, p2, p3]
host.legend(handles=lns, loc='best')

# right, left, top, bottom
par2.spines['right'].set_position(('outward', 60))
# no x-ticks
#par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
#par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')


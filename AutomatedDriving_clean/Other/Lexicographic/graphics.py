from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'arial',
        'size'   : 28}

matplotlib.rc('font', **font)


points = np.load('graphics.npy')

x_0 = points[:,0]
x_E = points[:,1]
x_E_2 = points[:,2]


mean_scores0 = list()
scores0 = deque(maxlen=500)
for point in x_0:
    scores0.append(point)
    mean_score = [np.mean(scores0)]
    mean_scores0.append(mean_score)

mean_scoresE = list()
scoresE = deque(maxlen=500)
for point in x_E:
    scoresE.append(point)
    mean_score = [np.mean(scoresE)]
    mean_scoresE.append(mean_score)

mean_scoresE2 = list()
scoresE2 = deque(maxlen=500)
for point in x_E_2:
    scoresE2.append(point)
    mean_score = [np.mean(scoresE2)]
    mean_scoresE2.append(mean_score)


fig, ax = plt.subplots()
plt.axhline(y=5.283)

plt.axhline(y=0.0)

n_episodes = len(points)
plt.plot(range(10,n_episodes), mean_scores0[10:n_episodes], markersize=10, marker='s', markevery=5000, label="Achievement $V_a$", color='red') # plotting t, a separately
plt.plot(range(10,n_episodes), mean_scoresE[10:n_episodes], markersize=10, marker='^', markevery=5000, label="Comfort $V_c$", color='green')
plt.plot(range(10,n_episodes), mean_scoresE2[10:n_episodes], markersize=10, marker='^', markevery=5000, label="Safety $V_s$", color='blue')


ax.set(xlabel='Episode', ylabel='Discounted sum of rewards')
plt.legend(loc='best')
plt.show()
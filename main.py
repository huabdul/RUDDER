import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from model import create_model
plt.rcParams['figure.dpi'] = 170
#%% Generate Dataset
N = 10000
actions = 2 * np.random.randint(0, 2, (N, 50)) - 1
states = np.zeros((N, 50))

for i in range(states.shape[1]-1):
    a = actions[:, i]
    states[:, i+1] = np.clip(states[:, i] + a, -6, 6)

rewards = np.sum(states == 2, axis=1)
states = states / 6
#%%
rnn = create_model()

epochs = 1000
batch_size = 20
losses = []

for e in range(epochs):
    for i in range(states.shape[0] // batch_size - 1):
        
        with tf.GradientTape(persistent=True) as tape:
            
            s = states[i*batch_size:(i+1)*batch_size, :]
            a = actions[i*batch_size:(i+1)*batch_size, :]
            r = rewards[i*batch_size:(i+1)*batch_size]
            s = np.expand_dims(s, -1)
            a = np.expand_dims(a, -1)
            r = np.array(r)
            
            pred = rnn([s, a])
            
            main_loss = tf.reduce_mean((r - pred[:, -1, 0])**2)
            aux_loss = tf.reduce_mean((r[:, None] - pred[:, :, 0])**2)
            loss = main_loss + aux_loss

        # optimizer = RMSprop(0.0001, centered=True)
        optimizer = Adam(0.001)

        gradients = tape.gradient(loss, rnn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rnn.trainable_variables))
        
        loss = loss.numpy()
        losses.append(loss)
        
        print(f'Epoch: {e} -- Batch: {i} -- Loss: {loss:.4f}')

#%%
for _ in range(50):
    n = np.random.randint(N)
    
    ss = states[n, :][None, :]; aa = actions[n, :][None, :]
    ss = np.expand_dims(ss, -1); aa = np.expand_dims(aa, -1)
    preds = rnn([ss, aa]).numpy().squeeze()
    
    rr = preds[1:] - preds[:-1]
    rr = np.concatenate(([preds[0]], rr))
    
    pred_return = sum(rr)
    true_return = rewards[n]
    pred_error = true_return - pred_return
    rr += pred_error / rr.shape[0]
    
    # plt.plot(states[n, :])
    # plt.hlines(2/6, 0, 50, ls='--')
    direction = (-2 * (states[n, :] > 2/6) + 1) * actions[n, :]
    plt.plot(direction)
    plt.plot(rr)
    # orig_r = np.zeros(50); orig_r[-1] = rewards[n]
    # plt.plot(orig_r)
    plt.grid()
    plt.title(f'N = {n}. Total Reward = {rewards[n]}')
    plt.legend(['+1 towards goal, -1 away from goal', 'Redistributed reward', 'Original reward'])
    plt.show()
# plt.plot(states[n, :])
# plt.hlines(2/6, 0, 50, ls='--')
# direction = -1 * (states[n, :] > 2/6) * actions[n, :]
# plt.plot(direction)



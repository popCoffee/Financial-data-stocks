
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = data
first_data = data.shape[0]
set_two = data.shape[1]
data = np.array(data)

# Training and test data
train_end = int(np.floor(0.8* first_data))
train_start = 0
test_start = train_end
test_end = first_data
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

#format the data and scale
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]
##===================================

#parameters again
batch_size = 100
learning_rate = 0.002
training_epochs = 12
n_neurons_4 = 128
n_neurons_3 = 256
n_neurons_2 = 512
n_neurons_1 = 1024

n_target = 1
numb_stocks = X_train.shape[1]

# Make Session
net = tf.InteractiveSession()
# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, numb_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])
# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# the first layer: hidden weights and bias variables
W_hidden_1 = tf.Variable(weight_initializer([numb_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))
# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
# stock data needs output layer transposed
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
# set the cost(mse) and optimizer
mse = tf.reduce_mean(tf.squared_difference(out,Y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)



# Establish initializer
net.run(tf.global_variables_initializer())

# Get the interactive plot
plt.ion()
fig = plt.figure()
Bxi1 = fig.add_subplot(111)
plottingA, = Bxi1.plot(y_test)
plottingB, = Bxi1.plot(y_test*0.5)
plt.show()

# Number of cycles and batch size
cycles = 300
batch_size = 256
#fit the NN
mse_train = []
pred_test=[]
mse_test  = []
for e in range(cycles):

    # randomize the training data with a shuffle
    shuffle_mix = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_mix]
    y_train = y_train[shuffle_mix]
    
    # training the small batch
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
        # Show progress
        if (i%50000) == 0:
            
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            plottingB.set_ydata(pred)
            plt.pause(0.2)

print(mse_train)
# Print final MSE after Training. Should be an array of decreasing floats

print("\n end =")
plt.ion()
fig = plt.figure()
Bxi1 = fig.add_subplot(111)
plottingA, = Bxi1.plot(y_test)
plottingB, = Bxi1.plot(y_test*0.5)
plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
tf.reset_default_graph()
# Import MINST data
xlsfile=pd.read_excel('data.xlsx',header=None)
data=np.array(xlsfile).astype('float32')
n=np.random.randint(0,data.shape[0],data.shape[0])

#load data/features from numpy array
train_data = data[n[0:3000],0:8]
valid_data = data[n[3000:3500],0:8]
test_data = data[n[3500:],0:8]
#load labels from numpy array
train_label = data[n[0:3000],8]
valid_label = data[n[3000:3500],8]
test_label = data[n[3500:],8]

#fixed random seed
tf.set_random_seed(100)

# Parameters
learning_rate = 0.1
num_epochs=100
batch_size = 2

# Network Parameters
n_input = 8 # MNIST data input (img shape: 28*28)
n_steps = 1 # timesteps
n_hidden = 10 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None,n_steps, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x):   
    # Permuting batch_size and n_steps
#    x = tf.transpose(x, [1, 0, 2])
#    # Reshape to (n_steps*batch_size, n_input)
#    x = tf.reshape(x, [-1, n_input])
#    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#    x = tf.split(x, n_steps)
    x = tf.reshape(x , [-1, n_steps, n_input])

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    # Get lstm cell output

    outputs, _= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
    # fully connect
    output_sequence=tf.matmul(tf.reshape(outputs, [-1,2*n_hidden]), weights['out']) + biases['out']
    return tf.reshape(output_sequence, [-1, n_steps,n_classes])

pred = BiRNN(x)
# Define loss and optimizer
cost = tf.losses.mean_squared_error(predictions = pred, labels = y)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
                learning_rate,
                global_step,
                num_epochs, 0.99,
                staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-10).minimize(cost,global_step=global_step)


# Initializing the variables
init = tf.global_variables_initializer()


train = []
valid = []
test = []
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    N = train_data.shape[0]

    # training cycle
    for epoch in range(num_epochs):
        # Apply SGD, each time update with one batch and shuffle randomly
        total_batch = int(math.ceil(N / batch_size))
        indices = np.arange(N)
        np.random.shuffle(indices)
        avg_loss = 0

        for i in range(total_batch):
            #get one batch
            rand_index = indices[batch_size*i:batch_size*(i+1)]
            batch_x = train_data[rand_index,:];batch_x = batch_x.reshape(-1,n_steps, n_input)
            batch_y = train_label[rand_index];batch_y = batch_y.reshape(-1, n_steps, n_classes)
#            batch_x,batch_y=sess.run([batch_x,batch_y])
            loss,_ = sess.run([cost,optimizer],feed_dict={x: batch_x, y: batch_y})
            avg_loss += loss / total_batch
        
        avg_loss = np.sqrt(avg_loss)
        train.append(avg_loss)
    
        valid_data=valid_data.reshape(-1,n_steps,n_input)
        valid_y = valid_label.reshape(-1, n_steps, n_classes)
        valid_loss = sess.run(cost, feed_dict={x: valid_data, y: valid_y})
        acg_valid_loss=np.sqrt(valid_loss)
        valid.append(acg_valid_loss)
        print('epoch:',epoch,' ,train loss ',avg_loss,' ,valid loss: ',acg_valid_loss)
    print("Optimization Finished!")
    
    test_data=test_data.reshape(-1,n_steps,n_input)
    test_pred = sess.run(pred, feed_dict={x: test_data})
    test_pred = test_pred.reshape(-1, n_classes)
    
test_mse=np.mean(np.square(test_pred-test_label))


# In[] plot RMSE vs epoch

g = plt.figure(1)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.plot( train, label='training')
plt.plot( valid, label='validation')
plt.title('loss curve')
plt.legend()
plt.show()    
# plot test_set result
for i in range(4):
    plt.figure()
    plt.plot(test_label[:,i],c='r', label='true')
    plt.plot(test_pred[:,i],c='b',label='predict')
    string='the '+str(i+1)+' output'
    plt.title(string)
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
#implement LSTM with TensorFlow
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pandas as pd
tf.reset_default_graph()
# In[]
xlsfile=pd.read_excel('data.xlsx',header=None)
data=np.array(xlsfile)
n=np.random.randint(0,data.shape[0],data.shape[0])

#load data/features from numpy array
train_data = data[n[0:3000],0:8]
valid_data = data[n[3000:3500],0:8]
test_data = data[n[3500:],0:8]
#load labels from numpy array
train_label = data[n[0:3000],8:]
valid_label = data[n[3000:3500],8:]
test_label = data[n[3500:],8:]

#fixed random seed
tf.set_random_seed(100)

#define hyper params
num_epochs = 10
batch_size = 2
alpha = 0.01
hidden_nodes = 40

input_features = 8
sequence_len = 1
output_class = 4 #regression 8 input----4 output

# input placeholder
X = tf.placeholder("float", [None, sequence_len, input_features])
Y = tf.placeholder("float", [None, sequence_len, output_class])

# define weights, gaussian distribution
weights = {
    'out': tf.Variable(tf.random_normal([hidden_nodes, output_class]))
}

biases = {
    'out': tf.Variable(tf.random_normal([output_class]))
}

# define the RNN lstm network
def RNN(x):
    # reshape input tensor into batch x sequence length x # of features

    x = tf.reshape(x , [-1, sequence_len, input_features])

    # triple layer LSTM with same number of nodes each layer
#    lstm_cell1 = tf.nn.rnn_cell.LSTMCell(10)
#    lstm_cell2 = tf.nn.rnn_cell.LSTMCell(10)
    lstm_cell3 = tf.nn.rnn_cell.LSTMCell(hidden_nodes)

    #stack of those layers
#    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2, lstm_cell3])
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell3])

    #initialize state
    init_state = lstm_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

    #get the output of each state
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, initial_state=init_state)
    output_sequence = tf.matmul(tf.reshape(outputs, [-1, hidden_nodes]), weights['out']) + biases['out']

    return tf.reshape(output_sequence, [-1, sequence_len, output_class])


#initialization
logits = RNN(X)
loss = tf.losses.mean_squared_error(predictions = logits, labels = Y)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
                alpha,
                global_step,
                num_epochs, 0.99,
                staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-10).minimize(loss,global_step=global_step)
init = tf.global_variables_initializer()



#lists to keep track of training, validation and test loss at each epoch
train = []
valid = []
test = []

with tf.Session() as sess:
    sess.run(init)

    #get # of inputs
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
            x = train_data[rand_index]
            x = x.reshape(-1,sequence_len, input_features)
            y = train_label[rand_index]
            y = y.reshape(-1, sequence_len, output_class)
            _, cost = sess.run([optimizer, loss],
                                feed_dict={X: x, Y: y})
            avg_loss += cost / total_batch
            
        # Monitoring Validation and Test Loss per each epoch updates
        valid_data=valid_data.reshape(-1,sequence_len,input_features)
        valid_y = valid_label.reshape(-1, sequence_len, output_class)
        valid_loss = sess.run(loss, feed_dict={X: valid_data, Y: valid_y})

        #take the square root
        avg_loss = np.sqrt(avg_loss)
        valid_loss = np.sqrt(valid_loss)

        #append to list for drawing
        train.append(avg_loss)
        valid.append(valid_loss)
        print('epoch:',epoch,' ,train loss ',avg_loss,' ,valid loss: ',valid_loss)

    #calculate result each dataset
    train_data=train_data.reshape(-1,sequence_len,input_features)
    train_pred = sess.run(logits, feed_dict={X: train_data})
    train_pred = train_pred.reshape(-1, output_class)

    valid_data=valid_data.reshape(-1,sequence_len,input_features)
    valid_pred = sess.run(logits, feed_dict={X: valid_data})
    valid_pred = valid_pred.reshape(-1, output_class)
    
    test_data=test_data.reshape(-1,sequence_len,input_features)
    test_pred = sess.run(logits, feed_dict={X: test_data})
    test_pred = test_pred.reshape(-1, output_class)

# calculate mse

test_mse=np.mean(np.square(test_pred-test_label))


# In[] plot RMSE vs epoch

g = plt.figure(1)
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.plot( train, label='training')
#plt.plot( valid, label='validation')
#plt.plot( test, label='test')
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





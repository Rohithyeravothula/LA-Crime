import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



input_columns = ["Year", "Victim Age"]
y_column = "Weapon"

def one_hot_encoder(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode




def build_column(filename="../database.csv"):
    data = pd.read_csv(filename)
    y = one_hot_encoder(data[y_column])
    x = data[input_columns]
    return x, y

first_hidden_layer_neurons = 100
second_hidden_layer_neurons = 50
output_layer_neurons = 16
learning_rate = 0.03

x = tf.placeholder(tf.float32, [None, len(input_columns)])
y = tf.placeholder(tf.float32, [None, output_layer_neurons])


# layer 1
W1 = tf.Variable(tf.truncated_normal([len(input_columns), first_hidden_layer_neurons], stddev=0.03))
b1 = tf.Variable(tf.truncated_normal([first_hidden_layer_neurons]))


# layer 2
W2 = tf.Variable(tf.truncated_normal([first_hidden_layer_neurons, second_hidden_layer_neurons], stddev=0.03))
b2 = tf.Variable(tf.truncated_normal([second_hidden_layer_neurons]))

# output layer

W3 = tf.Variable(tf.truncated_normal([second_hidden_layer_neurons, output_layer_neurons], stddev=0.03))
b3 = tf.Variable(tf.truncated_normal([output_layer_neurons]))



# model

first_layer = tf.add(tf.matmul(x, W1), b1)
first_layer = tf.nn.sigmoid(first_layer)

second_layer = tf.add(tf.matmul(first_layer, W2), b2)
second_layer = tf.nn.sigmoid(second_layer)

output_layer = tf.add(tf.matmul(second_layer, W3), b3)
output_layer = tf.nn.sigmoid(output_layer)


cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(output_layer) + (1-y)*tf.log(1 - output_layer), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)





X, Y = build_column()
X, Y = shuffle(X,Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.2, random_state=100)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


for epoch in range(0, 10):
    sess.run(optimizer, feed_dict={x: train_x, y:train_y})
    cost = sess.run(cross_entropy, feed_dict={x: train_x, y: train_y})
    print(cost)





import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    tf.set_random_seed(42)
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# The operations for the discriminator
with tf.name_scope("discriminator") as discriminator_scope:
    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, 784], name="X")

    with tf.name_scope("layers"):
        with tf.name_scope("1"):
            D_W1 = tf.Variable(xavier_init([784, 128]), name="D_W1")
            D_b1 = tf.Variable(tf.zeros(shape=[128]), name="D_b1")

        with tf.name_scope("2"):
            D_W2 = tf.Variable(xavier_init([128, 1]), name="D_W2")
            D_b2 = tf.Variable(tf.zeros(shape=[1]), name="D_b2")

theta_D = [D_W1, D_W2, D_b1, D_b2]

# The operations for the generator
with tf.name_scope("generator") as generator_scope:
    with tf.name_scope("inputs"):
        Z = tf.placeholder(tf.float32, shape=[None, 100], name="Z")

    with tf.name_scope("layers"):
        with tf.name_scope("1"):
            G_W1 = tf.Variable(xavier_init([100, 128]), name="G_W1")
            G_b1 = tf.Variable(tf.zeros(shape=[128]), name="G_b1")
        with tf.name_scope("2"):
            G_W2 = tf.Variable(xavier_init([128, 784]), name="G_W2")
            G_b2 = tf.Variable(tf.zeros(shape=[784]), name="G_b2")

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


# Export the plots to tensorboard via image

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

with tf.name_scope(generator_scope):
    G_sample = generator(Z)  # transform sample to image
# This is why 2 discriminators are present
# First, pass X through the discriminator, then the generator sample
# See V(G,D)
with tf.name_scope(discriminator_scope):
    D_real, D_logit_real = discriminator(X)  # real image value
    # there are new operations initialized when invoking this
    D_fake, D_logit_fake = discriminator(G_sample)  # fake image value

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
# ones like: 1 = true data
with tf.name_scope("loss"):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), name="D_loss_real")
    # zeros like: 0 is generated / fake data
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), name="D_loss_fake")
    # The following OP is not present in the graph!
    # This is adding 2 scalars
    D_loss = tf.add(D_loss_real, D_loss_fake, "D_loss")
    # The generator wants to make the discriminator meet 1
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), name="G_loss")

# Do we need a global step? See https://www.tensorflow.org/get_started/mnist/mechanics#training
# global_step = tf.Variable(0, name='global_step', trainable=False)


# Define the solvers
# Are two solvers really necessary?
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Interesting: No growth in GPU consumption with a larger batch size
mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('/media/niklas/lin-win-hdd/nn_datasets/image/mnist', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

no_of_generator_samples = 16
# TODO: Sort the summaries in collections and merge the collections if needed, sth. like
# image_summary = tf.summary.merge_all("image")

summary_writer = tf.summary.FileWriter('./logs/var_names_and_extended_name_scope', sess.graph)

# Generate summaries
image_summary = tf.summary.image("generator", tf.reshape(G_sample, shape=(no_of_generator_samples, 28, 28, 1)),
                                 max_outputs=9,
                                 collections="image")
D_loss_summary = tf.summary.scalar("D_loss", D_loss, collections="loss")
G_loss_summary = tf.summary.scalar("G_loss", G_loss, collections="loss")



# TODO: Substitute this loop with the global_step variable
Z_samples = sample_Z(no_of_generator_samples, Z_dim)
for it in range(1000000):

    # Every 1000 steps,pass inputs to the generator and save the output images
    if it % 5000 == 0:
        # TODO: It might not be helpful to sample always different z vectors. Idea: investigate one z vector
        # over time
        samples, summary = sess.run([G_sample, image_summary], feed_dict={Z: Z_samples})
        summary_writer.add_summary(summary, it)
        # fig = plot(samples)
        # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        # i += 1
        # plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

    # Output loss
    # TODO: It should be somehow possible to export the "true" image and the generator-image side by side
    _, D_loss_curr, d_sum_res = sess.run([D_solver, D_loss, D_loss_summary], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr, g_sum_res = sess.run([G_solver, G_loss, G_loss_summary], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    # TODO: Fetch a "collection" of summaries here instead of two separate ones. Arrange more intelligent


    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
        summary_writer.add_summary(d_sum_res, it)
        summary_writer.add_summary(g_sum_res, it)
        summary_writer.flush()  # well, this is not enough...


# TODO: Export checkpoints
# TODO: idea for embedding: export generated images and true images (or their represenations in intermediate layers) and
# check out, which ones are aligned
# TODO: Use tf.train.supervisor or tf.train.saver

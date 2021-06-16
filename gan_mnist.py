

import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#np.random.seed(0)
#tf.set_random_seed(2)

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 100 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 900000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
NUM_CLASSES = 10
CLASS_ITERS = 2

lib.print_model_settings(locals().copy())

global_step = tf.Variable(0, trainable=False)
class_coeff = tf.train.exponential_decay(1., global_step, 30000, 1.2, True)
step_update = global_step.assign_add(1)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None, labels=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    assert labels is not None, 'labels is not provided for generator'

    noise = tf.concat( [noise, labels],1)
    output = lib.ops.linear.Linear('Generator.Input', 138, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 10, output)

    return tf.reshape(output, [-1])

def Classifier(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Classifier.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Classifier.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Classifier.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Classifier.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Classifier.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Classifier.Output', 4*4*4*DIM, NUM_CLASSES, output)

    return tf.reshape(output, [-1, NUM_CLASSES])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_labels = tf.random_uniform([BATCH_SIZE], 0, 10, dtype=tf.int64)
fake_onehot = tf.one_hot(fake_labels, depth=10, on_value=1., off_value=0., axis=-1, dtype=tf.float32)
fake_data = Generator(BATCH_SIZE, labels=fake_onehot)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)
fake_logits = Classifier(fake_data)
real_logits = Classifier(real_data)
correct = tf.equal(tf.argmax(fake_logits, 1), fake_labels)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')
class_params = lib.params_with_name('Classifier')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=fake_onehot))
    gen_cost += class_coeff*class_cost
    with tf.control_dependencies([step_update]):
        gen_train_op = tf.train.RMSPropOptimizer(
            learning_rate=5e-5
        ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)
    class_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(class_cost, var_list=class_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=fake_onehot))
    gen_cost += class_coeff*class_cost

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    with tf.control_dependencies([step_update]):
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)
    class_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(class_cost, var_list=class_params)


    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real, 
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.
    class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=fake_onehot))
    gen_cost += class_coeff*class_cost

    with tf.control_dependencies([step_update]):
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5
        ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)
    class_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(class_cost, var_list=class_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(50, 128)).astype('float32'))
fixed_labels = [tf.constant(i, dtype=tf.int32, shape=[5]) for i in range(10)]
fixed_labels = tf.concat(fixed_labels ,0)
fixed_labels = tf.one_hot(fixed_labels, depth=10, on_value=1., off_value=0., axis=-1, dtype=tf.float32)
fixed_noise_samples = Generator(50, noise=fixed_noise, labels=fixed_labels)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((50, 28, 28)), 
        'samples_{}.png'.format(frame)
    )

# For evaluating accuracy

def accuracy(images=None, labels=None):

    if images is None:
        acc_ = session.run(acc)
        return acc_
    else:
        counts = np.zeros((10,10))
        logits_ = session.run(real_logits, feed_dict={real_data: images})   
        pre_label = np.argmax(logits_, 1)

        for idx, pre in enumerate(pre_label):
            counts[pre][labels[idx]] += 1
        acc_ = np.sum(np.max(counts, 1))/len(pre_label)
        nmi = metrics.normalized_mutual_info_score(labels, pre_label)
        ari = metrics.adjusted_rand_score(labels, pre_label)
        return acc_,nmi,ari



# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    _class_coeff = 0.1
    for iteration in range(ITERS):
        start_time = time.time()

        if iteration > 0:
            _, _class_coeff = session.run([gen_train_op, class_coeff])

        if iteration == ITERS/3:
            CRITIC_ITERS *= 2
            CLASS_ITERS *= 2

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = gen.__next__()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)
        for i in range(CLASS_ITERS):
            _class_cost, _ = session.run(
                [class_cost, class_train_op],
            )

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('train class cost', _class_cost)
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('class cost coefficient', _class_coeff)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            dev_class_accuracy = []
            dev_NMI = []
            dev_ARI = []
            dev_acc_2 = []
            for images,targets in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
                ACC,NMI,ARI=accuracy(images, targets)
                dev_class_accuracy.append(ACC)
                dev_NMI.append(NMI)
                dev_ARI.append(ARI)
            lib.plot.plot('dev fake class accuracy', accuracy())
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            lib.plot.plot('dev class accuracy', np.mean(dev_class_accuracy))
            lib.plot.plot('dev_NMI', np.mean(dev_NMI))
            lib.plot.plot('dev_ARI', np.mean(dev_ARI))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()

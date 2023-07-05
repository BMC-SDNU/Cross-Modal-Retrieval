import time
import os
import numpy as np
import tensorflow as tf
from utils import metrics
from coco import input_rank_coco as my_input
from coco.args import opt


if opt.dir == 'txt2img':
    y_dim = opt.txt_dim    # condition
    x_dim = opt.img_dim
else:  # img2txt
    y_dim = opt.img_dim
    x_dim = opt.txt_dim


def generator(y, reuse=None, is_training=False):
    G_prob = tf.layers.dense(y, x_dim,
                             kernel_initializer=tf.constant_initializer(np.eye(y_dim)),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=opt.wd),
                             bias_initializer=tf.constant_initializer(0.0),
                             name='fc1', reuse=reuse)
    return G_prob


def discriminator(x, y, reuse=None):
    # bilinear form
    x = tf.expand_dims(x, 2)
    y = tf.expand_dims(y, 2)
    inputs = tf.matmul(x, y, transpose_b=True)
    inputs = tf.reshape(inputs, [-1, y_dim * opt.h_dim])
    D_logit = tf.layers.dense(inputs, 1,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=opt.wd),
                              name='fc1', reuse=reuse)
    return D_logit


def run_training():
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        y = tf.placeholder(tf.float32, shape=[None, y_dim], name='condition')
        x = tf.placeholder(tf.float32, shape=[None, x_dim], name='target')
        is_training = tf.placeholder(tf.bool, shape=[])

        y1, y2, y3 = tf.split(y, 3, 0)
        x1, x2, x3 = tf.split(x, 3, 0)

        with tf.variable_scope('Gen') as scope:
            G_sample = generator(y1, reuse=None, is_training=is_training)
            scope.reuse_variables()
            G_representation = generator(y, reuse=True, is_training=is_training)
        with tf.variable_scope('Disc') as scope:
            D_real = discriminator(x1, y1, reuse=None)
            scope.reuse_variables()
            D_fake = discriminator(G_sample, y1, reuse=True)

            D_real2 = discriminator(x1, y2, reuse=True)
            D_fake2 = discriminator(G_sample, y2, reuse=True)

            D_real3 = discriminator(x2, y1, reuse=True)
            D_fake3 = discriminator(G_sample, y1, reuse=True)

            D_consx = discriminator(x3, y1, reuse=True)
            D_consy = discriminator(x1, y3, reuse=True)

        # loss of discriminator
        loss_d = 1*(-tf.reduce_mean(D_real) + tf.reduce_mean(D_fake)) + \
                 opt.beta*(1*tf.reduce_mean(D_consx) + 1*tf.reduce_mean(D_consy)) + \
                 1 *(-tf.reduce_mean(D_real2) + tf.reduce_mean(D_fake2)) + \
                 1 * (-tf.reduce_mean(D_real3))
        # loss of generator
        loss_g = -1*tf.reduce_mean(D_fake) - 1*tf.reduce_mean(D_fake2)

        # loss of MSE
        loss_eq = tf.nn.l2_loss(G_sample-x1)

        l2_loss = tf.losses.get_regularization_loss()

        tf.summary.histogram("D_real", D_real)
        tf.summary.histogram("D_fake", D_fake)
        tf.summary.histogram("G_sample", G_sample)

        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
        opt_d = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss_d, var_list=d_params)
        opt_g = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss_g+opt.gamma*loss_eq+l2_loss, var_list=g_params)
        clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_params]

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

    train_data_set, test_data_set = my_input.get_data(tt='cca')

    # Begin training.
    with tf.Session(graph=graph) as sess:

        # We must initialize all variables before we use them.
        sess.run(init)
        print("Initialized")

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(opt.log_dir, sess.graph)

        average_loss_d = 0
        average_loss_g = 0
        start_time = time.time()
        time_sum = 0
        with open(os.path.join(opt.log_dir, 'record.txt'), 'a') as f:
            f.write('-'*30+'\n')

        for step in range(opt.max_steps):
            # update discriminator
            for _ in range(opt.critic_itrs):
                batch_image, batch_text, batch_label = train_data_set.next_batch(opt.batch_size)
                if opt.dir == 'txt2img':
                    _, loss_val_d = sess.run([opt_d, loss_d],
                                             feed_dict={is_training: True, y: batch_text, x: batch_image})
                else:
                    _, loss_val_d = sess.run([opt_d, loss_d],
                                             feed_dict={is_training: True, y: batch_image, x: batch_text})
                sess.run(clip_d)
                average_loss_d += loss_val_d

            # update generator
            batch_image, batch_text, batch_label = train_data_set.next_batch(opt.batch_size)
            if opt.dir == 'txt2img':
                _, loss_val_g = sess.run([opt_g, loss_g],
                                         feed_dict={is_training: True, y: batch_text, x: batch_image})
            else:
                _, loss_val_g = sess.run([opt_g, loss_g],
                                         feed_dict={is_training: True, y: batch_image, x: batch_text})
            average_loss_g += loss_val_g

            # Write the summaries and print an overview fairly often.
            if (step + 1) % opt.log_interval == 0:
                average_loss_d /= (opt.log_interval * opt.critic_itrs)
                average_loss_g /= opt.log_interval
                duration = time.time() - start_time
                print('Step %d: average_loss_d = %.5f, average_loss_g = %.5f (%.3f sec)' %
                      (step, average_loss_d, average_loss_g, duration))
                average_loss_d = 0
                average_loss_g = 0

                if opt.dir == 'txt2img':
                    summary_str = sess.run(summary_op, feed_dict={y: batch_text, x: batch_image})
                else:
                    summary_str = sess.run(summary_op, feed_dict={y: batch_image, x: batch_text})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % opt.save_interval == 0 or (step + 1) == opt.max_steps:
                save_path = saver.save(sess, os.path.join(opt.save_dir, "model"), step)
                print("Model saved in file: %s" % save_path)
                batch_image, batch_text, batch_label = test_data_set.next_batch()
                if opt.dir == 'txt2img':
                    feed_dict = {is_training: False, y: batch_text, x: batch_image}
                    [text_representation, image_representation] = sess.run([G_representation, x], feed_dict=feed_dict)
                else:
                    feed_dict = {is_training: False, y: batch_image, x: batch_text}
                    [image_representation, text_representation] = sess.run([G_representation, x], feed_dict=feed_dict)
                duration = time.time() - start_time
                time_sum = time_sum + duration
                map_i2t, map_t2i = metrics.evaluate(image_representation, text_representation, batch_label, metric='cos')
                # np.savez("coco_pre_%d.npz" % (step + 1), img_proj_te=image_representation, txt_proj_te=batch_text,
                #          label_te=batch_label)
                start_time = time.time()
                with open(os.path.join(opt.log_dir, 'record.txt'), 'a') as f:
                    f.write('%d %f %f %.3f\n'% (step + 1, map_i2t[-1], map_t2i[-1], time_sum))


def main(_):
    if tf.gfile.Exists(opt.log_dir):
        tf.gfile.DeleteRecursively(opt.log_dir)
    tf.gfile.MakeDirs(opt.log_dir)
    run_training()


if __name__ == "__main__":
    tf.app.run()

import tensorflow as tf
import numpy as np
import sys,time,os
import utils
import operations


class BEGAN():
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        real_image_queue = utils.image_list(self.args.data_sets)
        self.real_data_set = utils.read_image(real_image_queue, self.args)
        self.build_model()

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.args.z_dim], name='noise_z')
        tf.summary.histogram('z', self.z)
        self.g = self.generator(self.z, name='generator', reuse=False, use_bn=self.args.use_bn, is_training=True)
        tf.summary.image('Gen', self.g)
        self.generated_sample = self.generator(self.z, name='generator', reuse=True, use_bn=self.args.use_bn, is_training=False)
        self.real_embedding, self.real_ae, self.real_pixelloss = self.auto_encoder(self.real_data_set, name='auto_encoder', reuse=False, use_bn=self.args.use_bn)
        tf.summary.image('AE_real', self.real_ae)
        self.gen_embedding, self.gen_ae, self.gen_pixelloss = self.auto_encoder(self.g, name='auto_encoder', reuse=True, use_bn=self.args.use_bn)
        tf.summary.image('AE_gen', self.gen_ae)

        # For balancing generator & discriminator loss, Proportional control Theory
        # Trainable false since we will assign it
        self.kt = tf.Variable(initial_value=0, name='k_t', trainable=False, dtype=tf.float32)
        tf.summary.scalar('k_t', self.kt)

        self.d_loss = self.real_pixelloss - self.kt * self.gen_pixelloss
        tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss = self.gen_pixelloss
        tf.summary.scalar('g_loss', self.g_loss)
        
        # Assign operation needs 'ref' to be a variable(mutable tensor)
        self.lr = tf.Variable(initial_value=self.args.learning_rate, name='learning_rate', trainable=False)
        # Initial learning rate : 0.0001, decaying by a factor of 2 when the measure of convergence stalls.
        self.lr_update = tf.assign(self.lr, tf.maximum(self.lr / 2, 0.00001))
        
        tr_vrbs = tf.trainable_variables()
        for i in tr_vrbs:
            print(i.name)           
        self.d_param = [v for v in tr_vrbs if v.name.startswith('auto_encoder')]
        self.g_param = [v for v in tr_vrbs if v.name.startswith('generator')] 
        d_optimizer = tf.train.AdamOptimizer(self.lr)
        d_grads = d_optimizer.compute_gradients(self.d_loss, var_list=self.d_param)
        for grad, var in d_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/gradient', grad)
        self.d_opt = d_optimizer.apply_gradients(d_grads)

        g_optimizer = tf.train.AdamOptimizer(self.lr)
        g_grads = g_optimizer.compute_gradients(self.g_loss, var_list=self.g_param)
        for grad, var in g_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/gradient', grad)
        self.g_opt = g_optimizer.apply_gradients(g_grads) 

        # k_t is between 0 and 1
        with tf.control_dependencies([self.d_opt, self.g_opt]):
            self.kt_update = tf.assign(self.kt, tf.clip_by_value(self.kt + self.args.lr_for_k*(self.args.gamma*self.real_pixelloss - self.gen_pixelloss), 0, 1))

        self.convergence_measure = self.real_pixelloss + tf.abs(self.args.gamma*self.real_pixelloss - self.gen_pixelloss)
        tf.summary.scalar('convergence_measure', self.convergence_measure)
    
        self.saver = tf.train.Saver()  

    def train(self):
        start_time = time.time()
        self.sess.run(tf.global_variables_initializer()) 
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.sample_z = np.random.uniform(-1, 1, [self.args.showing_height*self.args.showing_width, self.args.z_dim])
        sample_dir = os.path.join(self.args.sample_dir, self.model_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
   
        if self.load():
            print('Checkpoint loaded')
        else:
            print('Checkpoint load failed')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
     
        try:
            for epoch in range(self.train_count, self.args.num_epochs):
                print('Epoch %d starts' % (epoch+1))
                batch_z = np.random.uniform(-1, 1, [self.args.batch_size, self.args.z_dim])
                # theta_D, theta_G are updated independently based on their respective losses with seperate Adam
                disc_loss, gen_loss, _, _, c_measure = self.sess.run([self.d_loss, self.g_loss, self.d_opt, self.g_opt, self.convergence_measure], feed_dict={self.z:batch_z})
                # Update k_t at every training step
                self.sess.run(self.kt_update, feed_dict={self.z:batch_z})
                
                # Decaying learning rate
                if np.mod(epoch+1, self.args.lr_update_step) == 0:
                    self.sess.run(self.lr_update)

                if np.mod(epoch+1, self.args.log_step) == 0:
                    sum_op = self.sess.run(self.summary_op, feed_dict={self.z:batch_z})
                    self.summary_writer.add_summary(sum_op, epoch+1)

                if np.mod(epoch+1, self.args.save_step) == 0:
                    G_sample = self.sess.run(self.generated_sample, feed_dict={self.z:self.sample_z})
                    print(G_sample.shape)
                    utils.save_image(G_sample, [self.args.showing_height, self.args.showing_width], os.path.join(sample_dir, 'train_{:2d}steps.jpg'.format(epoch+1)))
                    self.save(global_step=epoch+1)
                print('Epoch %d, generator loss : %3.4f, discriminator loss : %3.4f, convergence : %3.4f, duration time : %3.4f' % (epoch+1, gen_loss, disc_loss, c_measure, time.time()-start_time))
        except tf.errors.OutOfRangeError:
            print('Epoch limited')
        except KeyboardInterrupt:
            print('End training')
        finally:
            coord.request_stop()
            coord.join(threads=threads)   

    
    def test(self):
        with tf.variable_scope('test'):
            self.z_r = tf.get_variable('z_r', [self.batch_size, self.z_dim], trainable=True, initializer=tf.trunated_normal_initializer(stddev=0.02))
        G_z_r = self.generator(self.z_r, name='generator', reuse=True, use_bn=self.args.use_bn, is_training=False)
        
        self.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runner(sess=self.sess, coord=coord)
        # Get just 1 batch, since we need to fix batch
        x = self.sess.run(self.real_data_sets)
        self.z_r_loss = tf.reduce_mean(tf.abs(x - G_z_r))
        self.z_r_opt = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.z_r_loss, var_list=[self.z_r])
        test_sample_dir = os.path.join(self.args.sample_dir, 'test')
        if not os.path.exists(test_sample_dir):
            os.mkdir(test_sample_dir)

        try:
            # Minimize e_r to find a value for z_r
            for steps in range(z_r_steps):
                z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_opt])
                print('Step %d, z_r_loss : %3.4f' %(steps, z_r_loss))
            # Get z
            z = self.sess.run(self.z_r)
            # [batch_size/2, 64, 64, 3]
            x_left, x_right = x[:int(self.args.batch_size/2)], x[int(self.args.batch_size/2):]
            z_left, z_right = z[:int(self.args.batch_size/2)], z[int(self.args.batch_size/2):]
           
            # Doing interpolation using spehrical interpolation
            # 0.1, 0.2 .... 0.9
            interpolated = list()
            for interpolate_idx, ratio in enumerate(np.linspace(0,1,11)):
                # For each image, doing interpolation
                # [batch_size2, 64, 64, 3]
                z = np.stack([operations.slerp(ratio, r1, r2) for r1, r2 in zip(z_left, z_right)])  
                G_z = self.generator(z, name='generator', reuse=True, use_bn=self.args.use_bn, is_training=False)
                interpolated.append(G_z)
            # interpolated : [11, batch_size/2, 64, 64, 3]
            # first and last element are real images
            interpolated.insert(0, x_left)
            interpolated.append(x_right)
            # Convert to numpy array to get shape, and arrange with batch order
            interpolated = np.asarray(interpolated).transpose([1,0,2,3,4])
            print('Image shape %s' % interpolated.shape)
            total_img_num = interpolated.shape[0] * interpolated.shape[1] # Will be batch size/2 * (11+2)
            total_image = tf.reshape(interpolated, [total_img_num] + interpolated.shape[2:])
            utils.save_image(total_image, [self.args.batch_size, 11+2], os.path.join(test_sample_dir, '{}_z_dim_total.jpg'.format(self.args.z_dim)))
        finally:
            coord.request_stop()
            coord.join(threads)


    def generator(self, z, is_training=True, name='generator', reuse=True, use_bn=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            generator_fc = operations.linear(z, self.args.target_size // 8 * self.args.target_size // 8 * self.args.filter_depth, name='gen_linear')
            x = tf.reshape(generator_fc, [-1, self.args.target_size // 8, self.args.target_size // 8, self.args.filter_depth])
            for i in range(4):
                gen_conv1 = operations.conv2d(x, self.args.filter_depth, filter_height=3, filter_width=3, stride_h=1, stride_v=1, use_bn=use_bn, name='gen_conv_%d1' % (i+1))
                gen_conv1_elu = operations.elu(gen_conv1, name='gen_conv_%d1_elu' % (i+1), is_training=is_training)
                gen_conv2 = operations.conv2d(gen_conv1_elu, self.args.filter_depth, filter_height=3, filter_width=3, stride_h=1, stride_v=1, use_bn=use_bn, name='gen_conv_%d2' % (i+1))
                gen_conv2_elu = operations.elu(gen_conv2, name='gen_conv_%d2_elu' % (i+1), is_training=is_training)
                if i < 3:
                    # Upsampling via nearest neighbor 
                    x = tf.image.resize_nearest_neighbor(gen_conv2_elu, size=(int(self.args.target_size//(2**(2-i))), int(self.args.target_size//(2**(2-i))))) 
                else:
                    x = gen_conv2_elu
            generator_result = operations.conv2d(x, self.args.num_channels, filter_height=3, filter_width=3, stride_h=1, stride_v=1, use_bn=use_bn, name='gen_conv_last')
            generator_result = operations.elu(generator_result, name='gen_conv_last_elu', is_training=is_training)
            return generator_result        

    
    def encoder(self, imgs, use_bn=False, is_training=True):
        with tf.variable_scope('encoder'):
            x = imgs # [batch, 64, 64, 3]
            for i in range(4):
                enc_conv1 = operations.conv2d(x, self.args.filter_depth*(i+1), filter_height=3, filter_width=3, stride_h=1, stride_v=1, name='enc_conv_%d1' % (i+1))
                enc_conv1_elu = operations.elu(enc_conv1, name='enc_conv_%d1_elu' % (i+1), is_training=is_training)
                enc_conv2 = operations.conv2d(enc_conv1_elu, self.args.filter_depth*(i+1), filter_height=3, filter_width=3, stride_h=1, stride_v=1, name='enc_conv_%d2' % (i+1))
                enc_conv2_elu = operations.elu(enc_conv2, name='enc_conv_%d2_elu' % (i+1), is_training=is_training)
                # Down sampling with strides 2
                if i < 3:
                    x = operations.conv2d(enc_conv2_elu, self.args.filter_depth*(i+2), filter_height=3, filter_width=3, stride_h=2, stride_v=2, name='enc_downsample_%d' % (i+1))
                    x = operations.elu(x, name='enc_downsample_%d_elu' % (i+1), is_training=is_training)
                else:
                    x = enc_conv2_elu
            final_shape = x.get_shape().as_list() 
            flattend_conv = tf.reshape(x, [-1, final_shape[1] * final_shape[2] * final_shape[3]])
            embedding = operations.linear(flattend_conv, self.args.embedding_size, name='enc_fc_layer')
            # This embedding tensor is mapped via fc not followed by any non-linearities
            return embedding 
            
    # Decoder use same architecture with generator
    def decoder(self, embedding, use_bn=False, is_training=True):
        with tf.variable_scope('decoder'):
            print('Embedding shape %s' % embedding.get_shape())
            embedding_fc = operations.linear(embedding, self.args.target_size // 8 * self.args.target_size // 8 * self.args.filter_depth, name='dec_linear')
            x = tf.reshape(embedding_fc, [-1, self.args.target_size // 8, self.args.target_size // 8, self.args.filter_depth])
            print('Shape %s' % x.get_shape())
            for i in range(4):
               enc_conv1 = operations.conv2d(x, self.args.filter_depth, filter_height=3, filter_width=3, stride_h=1, stride_v=1, use_bn=use_bn, name='dec_conv_%d1' % (i+1))
               enc_conv1_elu = operations.elu(enc_conv1, name='enc_conv_%d1_elu' % (i+1), is_training=is_training)
               enc_conv2 = operations.conv2d(enc_conv1_elu, self.args.filter_depth, filter_height=3, filter_width=3, stride_h=1, stride_v=1, use_bn=use_bn, name='dec_conv_%d2'% (i+1))
               enc_conv2_elu = operations.elu(enc_conv2, name='enc_conv_%d2_elu' % (i+1), is_training=is_training)
               if i != 3:
                   # Upsampling via nearest neighbor
          		   x = tf.image.resize_nearest_neighbor(enc_conv2_elu, size=(int(self.args.target_size//(2**(2-i))), int(self.args.target_size//(2**(2-i)))))
               else:
                   x = enc_conv2_elu
            decoder_result = operations.conv2d(x, self.args.num_channels, filter_height=3, filter_width=3, stride_h=1, stride_v=1, use_bn=use_bn, name='dec_conv_last')
            decoder_result = operations.elu(decoder_result, name='dec_conv_last_elu', is_training=is_training)
            return decoder_result 


    def auto_encoder(self, imgs, name='auto_encoder', reuse=False, use_bn=False, is_training=True):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            embedding = self.encoder(imgs, use_bn=use_bn, is_training=is_training)
            ae_result = self.decoder(embedding, use_bn=use_bn, is_training=is_training)
            print('Shape of each %s, %s' % (imgs.get_shape(), ae_result.get_shape()))
            pixelwise_loss = self.pixel_loss(imgs, ae_result)
            # Returns embedding, auto encoder results, Loss
            return embedding, ae_result, pixelwise_loss

    
    def pixel_loss(self, imgs, ae):
        diff = tf.abs(imgs - ae)
        if self.args.eta == 1:
            return tf.reduce_mean(diff)
        elif self.args.eta == 2:
            return tf.sqrt(tf.reduce_mean(tf.squares(diff)))
        else:
            raise ValueError('Ony supprot 1 or 2 eta')
   
    @property
    def model_dir(self):
        if self.args.use_bn:
            return '{}_batch_{}_z_dim_{}_bn'.format(self.args.batch_size, self.args.z_dim, 'CelebA')
        else:
            return '{}_batch_{}_z_dim_{}'.format(self.args.batch_size, self.args.z_dim, 'CelebA')

    def save(self, global_step):
        model_name='BEGAN'
        checkpoint_path = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        self.saver.save(self.sess, os.path.join(checkpoint_path, model_name), global_step=global_step) 
        print('Save checkpoint at %d steps' % global_step)

  
    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print('Checkpoint loaded at %d steps' % self.train_count)
            return 1
        else:
            self.train_count = 0
            return 0
            



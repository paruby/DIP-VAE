import tensorflow as tf
import numpy as np
import sys

class DIP_VAE(object):
    def __init__(self, type):
        self.z_dim = 10
        self.type = type
        if type == "i":
            self.lambda_d = 100
            self.lambda_od = 10
        elif type == "ii":
            self.lambda_d = 80
            self.lambda_od = 80

        self.data_train, self.data_test = self._data_init()
        self.input_ph, self.enc_mean, self.enc_logvar, self.z_sample, self.dec_mean, self.dec_stoch = self._autoencoder_init()
        self.recon_loss, self.auto_encoder_loss = self._loss_init()
        self.train_step = self._optimizer_init()

        self.sess = tf.Session()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())


    def train(self):
        print("Beginning training")
        it=0
        while it < 300000:
            it += 1
            self.sess.run(self.train_step, {self.input_ph: self.sample_minibatch()})

            if it % 100 == 0:
                batch = self.sample_minibatch()
                ae_train_loss = self.sess.run(self.auto_encoder_loss, {self.input_ph: batch})
                recon_train_loss = self.sess.run(self.recon_loss, {self.input_ph: batch})
                print("Iteration %i: \n    Autoencoder loss (train) %f\n    Reconstruction loss (train) %f" % (it, ae_train_loss, recon_train_loss), flush=True)
                print("Iteration %i: \n    Autoencoder loss (train) %f\n    Reconstruction loss (train) %f" % (it, ae_train_loss, recon_train_loss), flush=True, file=open(self.type + '_train.log','a'))

                ae_test_loss = self.sess.run(self.auto_encoder_loss, {self.input_ph: self.data_test[0:500]})
                recon_test_loss = self.sess.run(self.recon_loss, {self.input_ph: self.data_test[0:500]})
                print("    Autoencoder loss (test) %f\n    Reconstruction loss (test) %f" % (ae_test_loss, recon_test_loss), flush=True)
                print("    Autoencoder loss (test) %f\n    Reconstruction loss (test) %f" % (ae_test_loss, recon_test_loss), flush=True, file=open(self.type + '_train.log','a'))

            if it % 10000 == 0:
                model_path = self.type + "_checkpoints/model"
                save_path = self.saver.save(self.sess, model_path, global_step=it)
                print("Model saved to: %s" % save_path)
                print("Model saved to: %s" % save_path, file=open(self.type + '_train.log','a'))

    def load_latest_checkpoint(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.type + '_checkpoints'))

    def sample_minibatch(self, batch_size=64, test=False):
        if test is False:
            indices = np.random.choice(range(len(self.data_train)), batch_size, replace=False)
            sample = self.data_train[indices]
        elif test is True:
            indices = np.random.choice(range(len(self.data_test)), batch_size, replace=False)
            sample = self.data_test[indices]
        return sample

    def make_plots(self):
        pass

    def _data_init(self):
        # dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz must be in the root
        # folder. Find this here: https://github.com/deepmind/dsprites-dataset
        dataset_zip = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding='bytes')
        imgs = dataset_zip['imgs']
        imgs = imgs[:, :, :, None] # make into 4d tensor

        # 90% random test/train split
        n_data = len(imgs)
        np.random.shuffle(imgs)
        data_train = imgs[0 : (9*n_data)//10]
        data_test = imgs[(9*n_data)//10 : ]

        return data_train, data_test

    def _autoencoder_init(self):
        # make placeholder for feeding in data during training and evaluation
        input_ph = tf.placeholder(shape=[None, 64, 64, 1], dtype=tf.float32, name="input")
        # define the encoder network
        e_mean, e_logvar = self._encoder_init(input_ph)
        # reparameterisation trick
        eps = tf.random_normal(shape=tf.shape(e_mean))
        z_sample = e_mean + (tf.exp(e_logvar / 2) * eps)
        # define decoder network. d_stoch is decoding of random sample
        # from posterior, d_mean is decoding of mean of posterior
        d_stoch = self._decoder_init(inputs=z_sample)
        d_mean  = self._decoder_init(inputs=e_mean, reuse=True)

        return input_ph, e_mean, e_logvar, z_sample, d_mean, d_stoch

    def _encoder_init(self, inputs):
        with tf.variable_scope("encoder"):
            inputs_reshape = tf.reshape(inputs, shape=[-1] + [np.prod(inputs.get_shape().as_list()[1:])] )
            e_1 = tf.layers.dense(inputs=inputs_reshape, units=1200, activation=tf.nn.relu, name="e_1")
            e_2 = tf.layers.dense(inputs=e_1, units=1200, activation=tf.nn.relu, name="e_2")
            e_mean = tf.layers.dense(inputs=e_2, units=self.z_dim, name="e_mean")
            e_logvar = tf.layers.dense(inputs=e_2, units=self.z_dim, name="e_logvar")

        return e_mean, e_logvar

    def _decoder_init(self, inputs, reuse=False):
        with tf.variable_scope("decoder"):
            d_1 = tf.layers.dense(inputs=inputs, units=1200, activation=tf.nn.tanh, name="d_1", reuse=reuse)
            d_2 = tf.layers.dense(inputs=d_1, units=1200, activation=tf.nn.tanh, name="d_2", reuse=reuse)
            d_3 = tf.layers.dense(inputs=d_2, units=1200, activation=tf.nn.tanh, name="d_3", reuse=reuse)
            d_4 = tf.layers.dense(inputs=d_3, units=4096, name="d_4", reuse=reuse)
            d_out = tf.reshape(d_4, shape=[-1, 64, 64, 1])
        return d_out

    def _loss_init(self):
        ### Regulariser part of loss has two parts: KL divergence and DIP-VAE part
        ## KL part:
        KL_divergence = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(self.enc_logvar) - self.enc_logvar + self.enc_mean**2,axis=1) - self.z_dim)

        ## DIP-VAE part:
        if self.type == "i":
            # expectation of mu (mean of distributions)
            exp_mu = tf.reduce_mean(self.enc_mean, axis=0)

            # expectation of mu mu.tranpose
            mu_expand1 = tf.expand_dims(self.enc_mean, 1)
            mu_expand2 = tf.expand_dims(self.enc_mean, 2)
            exp_mu_mu_t = tf.reduce_mean( mu_expand1 * mu_expand2, axis=0)

            # covariance of model mean
            cov = exp_mu_mu_t - tf.expand_dims(exp_mu, 0) * tf.expand_dims(exp_mu, 1)
            diag_part = tf.diag_part(cov)
            off_diag_part = cov - tf.diag(diag_part)

            regulariser_od = self.lambda_od * tf.reduce_sum(off_diag_part**2)
            regulariser_d = self.lambda_d * tf.reduce_sum((diag_part - 1)**2)

            dip_vae_regulariser = regulariser_d + regulariser_od

            total_regulariser = KL_divergence + dip_vae_regulariser
        elif self.type == "ii":
            # See equation (5) in DIP-VAE paper. There are two terms, E(Cov(..))
            # and Cov(E(...)) which we compute here separately.
            ## E(Cov(...))
            sigma = tf.matrix_diag(tf.exp(self.enc_logvar))
            exp_cov = tf.reduce_mean(sigma, axis=0)

            ## Cov(E(...))
            # expectation of mu (mean of distributions)
            exp_mu = tf.reduce_mean(self.enc_mean, axis=0)
            # expectation of mu mu.tranpose
            mu_expand1 = tf.expand_dims(self.enc_mean, 1)
            mu_expand2 = tf.expand_dims(self.enc_mean, 2)
            exp_mu_mu_t = tf.reduce_mean( mu_expand1 * mu_expand2, axis=0)
            # covariance of model mean
            cov_exp = exp_mu_mu_t - tf.expand_dims(exp_mu, 0) * tf.expand_dims(exp_mu, 1)
            cov_z = cov_exp + exp_cov

            diag_part = tf.diag_part(cov_z)
            off_diag_part = cov_z - tf.diag(diag_part)

            regulariser_od = self.lambda_od * tf.reduce_sum(off_diag_part**2)
            regulariser_d = self.lambda_d * tf.reduce_sum((diag_part - 1)**2)

            dip_vae_regulariser = regulariser_d + regulariser_od

            total_regulariser = KL_divergence + dip_vae_regulariser

        ### Reconstruction loss is bernoulli
        im = self.input_ph
        im_flat = tf.reshape(im, shape=[-1, 64*64*1])
        logits = self.dec_stoch
        logits_flat = tf.reshape(logits, shape=[-1, 64*64*1])
        recon_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_flat,
                                                    labels=im_flat),
                                                    axis=1),
                                                    name="recon_loss")

        auto_encoder_loss = tf.add(recon_loss, total_regulariser, name="auto_encoder_loss")

        return recon_loss, auto_encoder_loss


    def _optimizer_init(self):
        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.auto_encoder_loss, var_list=enc_vars+dec_vars)

        return train_step


if __name__ == "__main__":
    type, mode = [sys.argv[1], sys.argv[2]]
    assert type in ["i", "ii"]
    vae = DIP_VAE(type)
    if mode == "train":
        vae.train()
    elif mode == "load":
        vae.load_latest_checkpoint()

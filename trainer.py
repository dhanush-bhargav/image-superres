import time
import numpy
import tensorflow as tf
import loss_calc
import os

def train_step(generator_model, generator_opt, discriminator_model, discriminator_opt, VGG_MODEL, true_lr, true_hr):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        false_hr = generator_model(true_lr, training=True)

        true_outputs = discriminator_model(true_hr, training=True)
        false_outputs = discriminator_model(false_hr, training=True)

        gen_loss = loss_calc.vgg_loss(VGG_MODEL, true_hr, false_hr, output_layer='block2_conv1') + (1e-3)*adv_loss(false_outputs)
        disc_loss = loss_calc.disc_loss(true_outputs, false_outputs)

    gen_grads = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_opt.apply_gradients(zip(gen_grads, generator_model.trainable_variables))
    discriminator_opt.apply_gradients(zip(disc_grads, discriminator_model.trainable_variables))

    return gen_loss

def train(generator_model, discriminator_model, VGG_MODEL, dataset, epochs=10):
    
    generator_opt = tf.keras.optimizers.Adam(1e-4)
    discriminator_opt = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt, discriminator_optimizer=discriminator_opt, generator=generator_model, discriminator=discriminator_model)

    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch: " + str(epoch + 1))

        batch_num = 0
        total_loss = tf.constant(0, dtype=tf.float32)

        for batch in dataset:
            loss = train_step(generator_model, generator_opt, discriminator_model, discriminator_opt, VGG_MODEL, batch[0], batch[1])
            total_loss = total_loss + loss
            batch_num = batch_num + 1
            print("", end="\r", flush=True)
            print("Batch: " + str(batch_num) + ", loss: " + str(loss.numpy()), end="")

        print(". Average loss: " + str((total_loss/batch_num).numpy()))

        if (epoch+1)%10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    return generator_model, discriminator_model
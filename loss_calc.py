import tensorflow as tf

def vgg_loss(true_hr, false_hr, VGG_MODEL, output_layer='block5_conv4'):
  
    output = VGG_MODEL.get_layer(output_layer).output
    vgg_model = Model([VGG_MODEL.input], output)

    true_fts = vgg_model(true_hr)
    false_fts = vgg_model(false_hr)

    loss = tf.math.reduce_sum(tf.math.square(tf.math.subtract(true_fts, false_fts)))
    loss = (loss*0.006)/(true_fts.shape[0]*true_fts.shape[1])

    return loss

def adv_loss(disc_output):

    loss = -tf.math.reduce_sum(tf.math.log(disc_output))

    return loss

def disc_loss(true_outputs, false_outputs):
    true_loss = tf.keras.losses.MeanSquaredError()(tf.ones_like(true_outputs), true_outputs)
    false_loss = tf.keras.losses.MeanSquaredError()(tf.ones_like(false_outputs), false_outputs)

    return (true_loss + false_loss)
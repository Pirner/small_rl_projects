import tensorflow as tf
import tensorflow.keras.backend as K


def masked_huber_loss(mask_value, clip_delta):
    def f(y_true, y_pred):
        error = y_true - y_pred
        cond = K.abs(error) < clip_delta
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = 0.5 * K.square(mask_true * (y_true - y_pred))
        linear_loss = mask_true * (clip_delta * K.abs(error) - 0.5 * (clip_delta ** 2))
        huber_loss = tf.where(cond, masked_squared_error, linear_loss)
        return K.sum(huber_loss) / K.sum(mask_true)
    f.__name__ = 'masked_huber_loss'
    return f

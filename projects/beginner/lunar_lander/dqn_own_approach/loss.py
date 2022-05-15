import tensorflow as tf
import tensorflow.keras.backend as k_backend


def masked_huber_loss(mask_value, clip_delta):
    def f(y_true, y_pred):
        error = y_true - y_pred
        cond = k_backend.abs(error) < clip_delta
        mask_true = k_backend.cast(k_backend.not_equal(y_true, mask_value), k_backend.floatx())
        masked_squared_error = 0.5 * k_backend.square(mask_true * (y_true - y_pred))
        linear_loss = mask_true * (clip_delta * k_backend.abs(error) - 0.5 * (clip_delta ** 2))
        huber_loss = tf.where(cond, masked_squared_error, linear_loss)
        return k_backend.sum(huber_loss) / k_backend.sum(mask_true)

    f.__name__ = 'masked_huber_loss'
    return f

import tensorflow as tf

# Orthogonal Regularizer
class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, beta=1e-4):
        self.beta = beta

    def __call__(self, w):
        if len(w.shape) == 4:
            w = tf.reshape(w, (w.shape[-1], -1))  # Reshape Conv2D filters (out_channels, -1)
        
        # Compute Gram matrix (W^T W)
        w_t_w = tf.matmul(w, w, transpose_b=True)
        
        # Identity matrix of same size as Gram matrix
        identity = tf.eye(w.shape[0])
        
        orth_penalty = tf.norm(w_t_w - identity, ord='fro', axis=(0, 1)) ** 2
        return self.beta * orth_penalty

    def get_config(self):
        return {"beta": self.beta}


# Focal Loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, logits):
        y_pred = tf.nn.softmax(logits)
        y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0 - 1e-5)

        focal_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        true_class_probs = tf.reduce_sum(y_pred * y_true, axis=-1)
        modulating_factor = tf.pow(1.0 - true_class_probs, self.gamma)

        focal_loss = self.alpha * modulating_factor * focal_loss
        return tf.reduce_mean(focal_loss)


# Combined Loss
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, base_loss_fn, focal_loss_fn, model, orthogonal_regularizer, name="combined_loss_2"):
        super().__init__(name=name)
        self.base_loss_fn = base_loss_fn
        self.model = model
        self.orthogonal_regularizer = orthogonal_regularizer
        self.focal_loss_fn = focal_loss_fn

    def call(self, y_true, y_pred):
        base_loss = self.base_loss_fn(y_true, y_pred)
        focal_loss = self.focal_loss_fn(y_true, y_pred)

        # Compute the orthogonality regularization loss for all layers
        orth_loss = 0.0
        for layer in self.model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                orth_loss += self.orthogonal_regularizer(layer.kernel)

        total_loss = 0.7 * base_loss + 0.3 * focal_loss + orth_loss
        return total_loss

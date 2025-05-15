from flax.nnx import jit, vmap, value_and_grad, one_hot
from optax.losses import softmax_cross_entropy


@jit
def train(optimizer, model, x, y, w):
    def loss(model):
        logits = vmap(model)(x)
        labels = one_hot(y, logits.shape[-1]) * w
        return softmax_cross_entropy(logits, labels).mean()

    loss, grads = value_and_grad(loss)(model)
    optimizer.update(grads)
    return loss

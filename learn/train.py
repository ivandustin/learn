from flax.nnx import jit, value_and_grad, vmap


@jit
def train(model, optimizer, x, y):
    def loss(model):
        yhat = vmap(model)(x)
        return ((yhat - y) ** 2).mean()

    loss, grads = value_and_grad(loss)(model)
    optimizer.update(grads)
    return loss

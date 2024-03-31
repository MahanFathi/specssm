import jax
import jax.numpy as jnp
import optax

def create_classification_loss():
    """Creates a loss function for classification tasks, that operates on batches."""

    def classification_loss_fn(model_outputs, targets):
        num_classes = model_outputs.shape[-1]   # might have to fix later
        one_hot_targets = jax.nn.one_hot(targets, num_classes)
        loss = jnp.sum(optax.softmax_cross_entropy(model_outputs, one_hot_targets))
        metrics = {
            'loss': loss,
        }
        return loss, metrics
    
    return classification_loss_fn
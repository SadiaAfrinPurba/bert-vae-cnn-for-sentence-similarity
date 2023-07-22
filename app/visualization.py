import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_embedding(bert_embedding, masks, epoch, batch_no, kl_coefficent, type="bert") -> None:
    saved_path = os.path.join("plots", type, kl_coefficent, f"epoch-{str(epoch + 1)}")
    os.makedirs(saved_path, exist_ok=True)
    dim_reducer = TSNE(n_components=2, perplexity=10.0)
    layer_averaged_hidden_states = torch.div(bert_embedding.sum(dim=1), masks.sum(dim=1, keepdim=True))
    print(f"Shape of layer_averaged_hidden_states: {layer_averaged_hidden_states.shape}")
    layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states.numpy())
    df = pd.DataFrame.from_dict(
        {'x': layer_dim_reduced_embeds[:, 0], 'y': layer_dim_reduced_embeds[:, 1]})

    sns.scatterplot(data=df, x='x', y='y')
    plt.savefig(os.path.join(saved_path, f"batch-{str(batch_no)}.png"), format='png', pad_inches=0)


def visualize_latent_vector_space(latent_vector_space, epoch, batch_count, kl_coefficent, type="latent_vector_space") -> None:
    saved_path = os.path.join("plots", type, kl_coefficent, f"epoch-{str(epoch + 1)}")
    os.makedirs(saved_path, exist_ok=True)
    dim_reducer = TSNE(n_components=2,  perplexity=10.0)
    latent_vector_space_reduced = dim_reducer.fit_transform(latent_vector_space.numpy())
    df = pd.DataFrame.from_dict(
        {'x': latent_vector_space_reduced[:, 0], 'y': latent_vector_space_reduced[:, 1]})
    sns.scatterplot(data=df, x='x', y='y')
    plt.savefig(os.path.join(saved_path, f"batch-{str(batch_count)}.png"), format='png', pad_inches=0)







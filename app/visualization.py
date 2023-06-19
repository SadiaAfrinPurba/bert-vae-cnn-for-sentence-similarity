import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_raw_bert_embedding(bert_embedding, masks, epoch, type="bert") -> None:
    saved_path = os.path.join("plots", type)
    os.makedirs(saved_path, exist_ok=True)
    dim_reducer = TSNE(n_components=2)
    layer_averaged_hidden_states = torch.div(bert_embedding.sum(dim=1), masks.sum(dim=1, keepdim=True))
    layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states.cpu().numpy())
    df = pd.DataFrame.from_dict(
        {'x': layer_dim_reduced_embeds[:, 0], 'y': layer_dim_reduced_embeds[:, 1]})

    sns.scatterplot(data=df, x='x', y='y')
    plt.savefig(os.path.join(saved_path, f"{str(epoch + 1)}.png"), format='png', pad_inches=0)







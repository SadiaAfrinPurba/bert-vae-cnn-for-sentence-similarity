import os.path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from app import optimizer
from app.datasets import tokenizer
from app.datasets.wiki_dataset import WikiDataset
from app.loss import calculate_vae_loss
from app.models import bert
from app.models.cnn_encoder_decoder import CNNEncoder, CNNDecoder
from app.models.vae import VAE

torch.cuda.empty_cache()


@dataclass
class Config:
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    latent_dim: int = 32
    max_length: int = 128
    training_file_path: str = os.path.join("data", "wiki1m_for_simcse.txt")
    is_taking_subset: bool = False
    data_subset_size: int = 5000


def start_training(config: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_tokenizer = tokenizer.get_bert_tokenizer()
    bert_model = bert.get_bert_model()
    bert_model = bert_model.to(device)

    input_dim = bert_model.config.hidden_size

    wiki_dataset = WikiDataset(file_path=config.training_file_path,
                               tokenizer=bert_tokenizer, max_length=config.max_length,
                               is_taking_subset=config.is_taking_subset, data_subset_size=config.data_subset_size)
    train_size = int(0.9 * len(wiki_dataset))
    test_size = len(wiki_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(wiki_dataset, [train_size, test_size], generator=generator)

    data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    encoder = CNNEncoder(input_dim=input_dim, latent_dim=config.latent_dim).to(device)
    decoder = CNNDecoder(latent_dim=config.latent_dim, output_dim=input_dim).to(device)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=config.latent_dim)
    vae = vae.to(device)

    adam_optimizer = optimizer.get_adam_optimizer(model=vae, lr=config.learning_rate)
    num_batches = len(data_loader) // config.batch_size

    print("==============Training Performance========================")
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        total_similarity = 0.0
        total_cosine_dist = 0.0

        for input_ids, attention_mask in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
                bert_embeddings = outputs.last_hidden_state.squeeze(0)
                bert_embeddings = bert_embeddings.to(device)  # Shape (64, 128, 768) (batch, max_length, embed_dim)

            adam_optimizer.zero_grad()

            x = bert_embeddings.permute(0, 2, 1)  # Reshape to (batch_size, embedding_dim, sequence_length)
            x_hat, mu, logvar = vae(x)

            # print(f"Generated Sentence: {bert_tokenizer.decode(x_hat.argmax(dim=-1)[0], skip_special_tokens=True)}")

            loss = calculate_vae_loss(x, x_hat, mu, logvar)
            similarity = torch.cosine_similarity(x_hat, x, dim=2).mean()
            cosine_dist = 1 - similarity

            loss.backward()
            adam_optimizer.step()

            total_loss += loss.item()
            total_similarity += similarity.item()
            total_cosine_dist += cosine_dist

        avg_loss = total_loss / num_batches
        avg_similarity = total_similarity / num_batches
        avg_cosine_dist = total_cosine_dist / num_batches

        print(
            f"Epoch {epoch + 1}/{config.num_epochs}, Avg VAE_CNN Loss: {avg_loss:.2f}, Avg Cosine Distance: {avg_cosine_dist:.2f}, Avg Cosine Similarity: {avg_similarity:.2f}")

    print("==============Evaluation Performance on 100 test data (never seen during training phase)========================")
    vae.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_similarity = 0.0
        total_cosine_dist = 0.0
        similarity_info = []
        sentences_info = []

        for input_ids, attention_mask in test_data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
                bert_embeddings = outputs.last_hidden_state.squeeze(0)
                bert_embeddings = bert_embeddings.to(device)  # Shape (64, 128, 768) (batch, max_length, embed_dim)

            x = bert_embeddings.permute(0, 2, 1)  # Reshape to (batch_size, embedding_dim, sequence_length)
            x_hat, mu, logvar = vae(x)

            original_sentence = bert_tokenizer.decode(x.argmax(dim=-1)[0], skip_special_tokens=True)
            generated_sentence = bert_tokenizer.decode(x_hat.argmax(dim=-1)[0], skip_special_tokens=True)

            print(f"Original Sentence: {original_sentence}")
            print(f"Generated Sentence: {generated_sentence}")

            loss = calculate_vae_loss(x, x_hat, mu, logvar)
            similarity = torch.cosine_similarity(x_hat, x, dim=2).mean()
            similarity_info.append(similarity)
            sentences_info.append([original_sentence, generated_sentence])
            cosine_dist = 1 - similarity

            total_loss += loss.item()
            total_similarity += similarity.item()
            total_cosine_dist += cosine_dist

        avg_loss = total_loss / num_batches
        avg_similarity = total_similarity / num_batches
        avg_cosine_dist = total_cosine_dist / num_batches

        similarity_with_sentence_info = list(zip(similarity_info, sentences_info))
        similarity_with_sentence_info.sort()

        print(
            f"Avg VAE_CNN Loss: {avg_loss:.2f}, Avg Cosine Distance: {avg_cosine_dist:.2f}, Avg Cosine Similarity: {avg_similarity:.2f}")

        print("=============10 generated sentences with the highest cosine similarity=======")
        print("=========format ('similarity score, [original sent, generated sent]')=======================")
        for item in similarity_with_sentence_info[:10]:
            print(item)


if __name__ == '__main__':
    config = Config()
    start_training(config=config)

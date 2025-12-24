import random
from logging import getLogger

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset

logger = getLogger()


class TextJEPA(Dataset):
    """
    Text dataset for Text-JEPA.
    Loads text from HuggingFace datasets (TinyStories).
    """

    def __init__(self, dataset_name='roneneldan/TinyStories', split='train', transform=None, max_length=None):
        """
        :param dataset_name: Name of HuggingFace dataset
        :param split: 'train' or 'validation'
        :param transform: optional function to apply to text (e.g., tokenization)
        :param max_length: optionally truncate text
        """
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f'Loaded {len(self.dataset)} samples from {dataset_name} ({split})')

        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # The TinyStories dataset has 'text' field
        text = self.dataset[idx]['text']

        # Optional truncation
        if self.max_length is not None:
            text = text[:self.max_length]

        # Optional transform (tokenization, embedding, etc.)
        if self.transform:
            text = self.transform(text)

        return text


def make_textjepa(
    batch_size=32,
    collator=None,
    num_workers=4,
    dataset_name='roneneldan/TinyStories',
    split='train',
    transform=None,
    fraction=None,
    max_length=None
):
    """
    Create DataLoader for Text-JEPA
    :param fraction: optionally use only a fraction of data
    """
    dataset = TextJEPA(dataset_name=dataset_name, split=split, transform=transform, max_length=max_length)

    # Use fraction if specified
    if fraction is not None and fraction < 1.0:
        num_samples = int(len(dataset) * fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, indices)
        logger.info(f'Using {num_samples} samples ({fraction*100:.1f}% of dataset)')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    logger.info('Text-JEPA DataLoader created')
    return dataset, data_loader


# Example collator for Text-JEPA: simply returns list of strings
def simple_collator(batch):
    return batch


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(text):
        return tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')['input_ids'].squeeze(0)

    # Create dataset and dataloader
    dataset, data_loader = make_textjepa(
        batch_size=16,
        collator=simple_collator,
        transform=tokenize,
        fraction=0.1,  # use 10% for quick test
        max_length=128
    )

    # Print sample batch
    batch = next(iter(data_loader))
    print(f"Batch size: {len(batch)}")
    print(f"First tokenized sample shape: {batch[0].shape}")

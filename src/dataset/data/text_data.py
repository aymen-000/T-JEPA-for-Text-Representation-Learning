import random
from logging import getLogger
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, interleave_datasets
from src.dataset.masks.all_masks import TextMutiBlockMaskCollector

logger = getLogger()


class TextJEPA(Dataset):
    """
    Text dataset for Text-JEPA.
    Supports Wikipedia, BookCorpus, C4, or any HuggingFace text dataset.
    Recommended for SSL pretraining + linear probing.
    """

    def __init__(
        self,
        dataset_name='wikimedia/wikipedia',
        dataset_config='20231101.en',
        split='train',
        text_field='text',          # 'text' for wiki/books, 'content' for c4
        transform=None,
        max_length=None,
    ):
        """
        :param dataset_name:   HuggingFace dataset repo name
        :param dataset_config: Dataset config/subset (e.g. '20231101.en' for Wikipedia)
        :param split:          'train' or 'validation'
        :param text_field:     Name of the text column in the dataset
        :param transform:      Optional callable (e.g. tokenizer)
        :param max_length:     Optional character-level truncation before transform
        """
        if dataset_config:
            self.dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            self.dataset = load_dataset(dataset_name, split=split)

        self.text_field = text_field
        self.transform = transform
        self.max_length = max_length
        logger.info(
            f'Loaded {len(self.dataset)} samples from '
            f'{dataset_name}/{dataset_config} ({split})'
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_field]

        # Strip very short samples that tokenize to near-empty sequences
        if len(text.strip()) < 20:
            text = "[PAD]"

        if self.max_length is not None:
            text = text[:self.max_length]

        if self.transform:
            text = self.transform(text)

        return text


class InterleavedTextJEPA(Dataset):
    """
    Interleaves Wikipedia + BookCorpus for richer pretraining signal.
    Mimics the original BERT pretraining data mixture.
    """

    def __init__(
        self,
        split='train',
        transform=None,
        max_length=None,
        seed=42,
    ):
        wiki = load_dataset('wikimedia/wikipedia', '20231101.en', split=split)
        books = load_dataset('bookcorpusopen', 'plain_text', split=split)

        # Interleave with equal probability
        self.dataset = interleave_datasets(
            [wiki, books],
            probabilities=[0.5, 0.5],
            seed=seed,
            stopping_strategy='first_exhausted',
        )
        self.text_field_map = {0: 'text', 1: 'text'}  # both use 'text'
        self.transform = transform
        self.max_length = max_length
        logger.info(f'Interleaved dataset size: {len(self.dataset)}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample.get('text', '')

        if len(text.strip()) < 20:
            text = "[PAD]"
        if self.max_length is not None:
            text = text[:self.max_length]
        if self.transform:
            text = self.transform(text)
        return text


def make_textjepa(
    batch_size=32,
    collator=None,
    num_workers=4,
    dataset_name='wikimedia/wikipedia',
    dataset_config='20231101.en',
    split='train',
    text_field='text',
    transform=None,
    fraction=None,
    max_length=None,
    use_interleaved=False,          # True for Wiki + BookCorpus mix
):
    """
    Create DataLoader for Text-JEPA SSL pretraining.

    Recommended dataset choices:
      - Wikipedia only:        dataset_name='wikimedia/wikipedia', dataset_config='20231101.en'
      - C4 (large scale):      dataset_name='c4',                 dataset_config='en'
      - Wiki + Books mixed:    use_interleaved=True
    """
    if use_interleaved:
        dataset = InterleavedTextJEPA(split=split, transform=transform, max_length=max_length)
    else:
        dataset = TextJEPA(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            text_field=text_field,
            transform=transform,
            max_length=max_length,
        )

    if fraction is not None and fraction < 1.0:
        num_samples = int(len(dataset) * fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, indices)
        logger.info(f'Using {num_samples} samples ({fraction * 100:.1f}% of dataset)')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    logger.info('Text-JEPA DataLoader created')
    return data_loader, dataset


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from src.dataset.masks.all_masks import TextMutiBlockMaskCollector

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(text):
        return tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
        )['input_ids'].squeeze(0)

    mask_collator = TextMutiBlockMaskCollector(
        max_tokens=128,
        nenc=1,
        npred=2,
        enc_mask_scale=[0.65, 0.85],
        pred_mask_scale=[0.10, 0.25],
        min_keep=4,
        allow_overlap=False,
    )

    # --- Option A: Wikipedia only (recommended starting point) ---
    loader, dataset = make_textjepa(
        batch_size=16,
        collator=mask_collator,
        transform=tokenize,
        fraction=0.01,           # 1% of Wikipedia ≈ 60k articles, plenty for a quick test
        max_length=512,          # character limit before tokenization; wiki articles are long
        dataset_name='wikimedia/wikipedia',
        dataset_config='20231101.en',
    )

    # --- Option B: Wikipedia + BookCorpus interleaved (BERT-style) ---
    # loader, dataset = make_textjepa(
    #     batch_size=16,
    #     collator=mask_collator,
    #     transform=tokenize,
    #     fraction=0.01,
    #     use_interleaved=True,
    # )

    # --- Option C: C4 for large-scale training ---
    # loader, dataset = make_textjepa(
    #     batch_size=16,
    #     collator=mask_collator,
    #     transform=tokenize,
    #     fraction=0.001,          # C4 is huge; even 0.1% is millions of samples
    #     dataset_name='c4',
    #     dataset_config='en',
    #     text_field='text',
    # )

    batch = next(iter(loader))
    tokens, masks_enc, masks_pred = batch
    print("mask_enc  ========>", masks_enc[0])
    print("mask_pred ========>", masks_pred[0])
    print("tokens    ========>", tokens[0])
    print("Shape:", tokens.shape, "| nenc:", len(masks_enc), "| npred:", len(masks_pred))
# Dataset Directory

This directory is the intended location for raw data assets used during
training and local testing.

## Download

The Flickr8k dataset (images + captions + tokenizer + model weights) is hosted on Google Drive:

🔗 **[Download Dataset & Weights](https://drive.google.com/drive/folders/1Q_NAnfN1bQWna2wY7wT8DinYlJe0Dtuk?usp=sharing)**

## After Downloading

1. Place model weights in `models/` (not `data/`):
   ```
   models/
   ├── modelConcat_1_89.h5
   └── caption_train_tokenizer.pkl
   ```

2. (Optional) Copy a few sample images to `data/samples/` for quick CLI testing:
   ```
   data/
   └── samples/
       ├── dog_park.jpg
       └── city_street.jpg
   ```

See [`docs/dataset.md`](../docs/dataset.md) for the full dataset description.

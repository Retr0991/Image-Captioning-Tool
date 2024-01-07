from PIL import Image
import torch, os
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Setting Dataset Path here 
dataset_path_ann = r'C:\Users\Retr0991\ML stuf\Project_IEEEMegaProj23\dataset\captions.txt'
dataset_path_img = r'C:\Users\Retr0991\ML stuf\Project_IEEEMegaProj23\dataset\Images'


# build the vocabulary from the captions
def build_vocabulary(caption_list):
    tokenized_captions = []
    # create tokenizer
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    # store tokens in a list of list
    for caption in caption_list:
        for word in tokenizer(caption.lower()):
            tokenized_captions.append(word)
    
    # return iterator          
    def yield_tokens(tokenized_captions):
        for word in tokenized_captions:
            yield word    
    
    # build vocabulary  
    vocab = build_vocab_from_iterator([yield_tokens(tokenized_captions)], specials=["<pad>", "<start>", "<end>", "<unk>"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = vocab.to(device)

    # string to index and index to string (don't use)
    stoi = {word: idx for word, idx in enumerate(vocab.get_stoi())}
    itos = {idx: word for idx, word in enumerate(vocab.get_itos())}
    return vocab

# Create Dataset
class FlickrDataset(Dataset):
    def __init__(self, image_directory, captions_file, captions_dict, transform):
        self.image_directory = image_directory
        self.captions_dict = captions_dict
        with open(captions_file) as f:
            self.captions = f.read().splitlines()
        self.image_filenames = os.listdir(self.image_directory)
        self.transform = transform
        self.vocab = build_vocabulary(self.captions)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_directory, image_filename)

        # Load image
        img = Image.open(image_path).convert("RGB")

        # Apply transformations if specified
        if self.transform is not None:
            img = self.transform(img)

        # Get corresponding captions
        captions = self.captions_dict[image_filename]
        captions = '<start> ' + ' '.join(captions) + ' <end>'
        captions, _ = convert_tokens(captions)
        return img, torch.tensor(captions)
    

# padding the images to be of the same length
class MyCollate:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = torch.tensor([item[1] for item in batch])
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_index)

        return images, targets


def convert_tokens(captions_raw):
    '''
    Converts raw text captions to indexed tokens using a vocabulary\n
    Takes a list of raw text captions.
    Tokenizes each caption using spaCy.
    Adds start and end tokens to each caption.
    Looks up index of each token in the vocabulary.\n
    Returns a list of caption token indexes and the vocabulary\n
    '''
    captions = []
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # convert raw tokens to indexes
    vocabulary = build_vocabulary(captions_raw)
    stoi = vocabulary.get_stoi()
    for caption in captions_raw:
        temp = []
        tokens = ['<start>'] + tokenizer(caption.lower()) + ['<end>']
        for token in tokens:
            temp.append(stoi[token])
        captions.append(temp)
    return captions, vocabulary


def get_loader(
        image_directory,
        annotation_file,
        captions_dict,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    '''
    Returns a DataLoader for the Flickr image captioning dataset.
        
        Args:
          image_directory: Path to the folder containing the images.
          annotation_file: Path to the CSV file containing the image filenames and captions.  
          captions_dict: Dictionary mapping image filenames to lists of captions.
          transform: Transformations to apply to each image.
          batch_size: Batch size for the DataLoader.
          num_workers: Number of worker processes for DataLoader. 
          shuffle: Whether to shuffle the dataset.
          pin_memory: Whether to pin memory in DataLoader.
          
        Returns:
          loader: DataLoader for the dataset.
          dataset: The FlickrDataset instance.
    '''
    dataset = FlickrDataset(image_directory, annotation_file, captions_dict, transform=transform)

    pad_index = dataset.vocab.get_stoi()["<pad>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=False,
        # collate_fn=MyCollate(pad_index=pad_index),
    )
    return loader, dataset


# import pandas as pd
# df = pd.read_csv(dataset_path_ann)
# y_train, voc = convert_tokens(["There's a cat on the mat.", "The dog is under the table."])

# for sent in y_train:
#     for word in sent:
#         print(f'{voc.get_itos()[word]} : {word}', end = ', ')
#     print()
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from caption_processing import get_loader
from model import CNNtoRNN


# Setting Dataset Path here 
dataset_path_ann = r'C:\Users\Retr0991\ML stuf\Project_IEEEMegaProj23\dataset\captions.txt'
dataset_path_img = r'C:\Users\Retr0991\ML stuf\Project_IEEEMegaProj23\dataset\Images'


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    df = pd.read_csv(dataset_path_ann)
    image_cap_dict = {}
    for i in range(df.shape[0]):
        if df.iloc[i, 0] not in image_cap_dict:
            image_cap_dict[df.iloc[i,0]] = [df.iloc[i,1]]
        else:
            image_cap_dict[df.iloc[i, 0]].append(df.iloc[i, 1])

    train_loader, dataset = get_loader(
        image_directory=dataset_path_img,
        annotation_file=dataset_path_ann,
        captions_dict=image_cap_dict,
        transform=transform,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 5

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.get_stoi()["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for _ in range(num_epochs):

        for _, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            # writer.add_scalar("Training loss", loss.item(), global_step=step)
            # step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()



if __name__ == "__main__":
    train()
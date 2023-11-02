import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from load_model import load_model
from dataset import BertDataset
from tqdm import tqdm


def train_epoch(train_loader: DataLoader, model, criterion, optimizer, device):

    # switch to training mode
    model.train()

    avg_loss = 0
    for data_batch in tqdm(train_loader):
        # clear old gradients
        optimizer.zero_grad()

        # split the batch into data ids, attention mask, and targets
        data_ids, attention_mask = data_batch['ids'], data_batch['mask']
        label, token_types = data_batch['target'], data_batch['token_type_ids']

        # use the gpu
        data_ids, attention_mask = data_ids.to(device), attention_mask.to(device)
        label, token_types = label.to(device), token_types.to(device)

        # forward the input through the model
        output = model(input_ids=data_ids, attention_mask=attention_mask, token_type_ids=token_types)

        # compute the loss
        loss = criterion(output['logits'], label)
        avg_loss += loss.detach().cpu().item()

        # back propagate the error (compute gradients)
        loss.backward()

        # update the parameters
        optimizer.step()
    avg_loss /= len(train_loader)
    return avg_loss


def validation_epoch(validation_loader: DataLoader, model, criterion, device) -> float:

    # switch to evaluation mode
    model.eval()

    avg_loss = 0
    with torch.no_grad():
        for data_batch in tqdm(validation_loader):
            try:

                # split the batch into data ids, attention mask, and targets
                data_ids, attention_mask = data_batch['ids'], data_batch['mask']
                label, token_types = data_batch['target'], data_batch['token_type_ids']

                # use the gpu
                data_ids, attention_mask = data_ids.to(device), attention_mask.to(device)
                label, token_types = label.to(device), token_types.to(device)

                # forward the input through the model
                output = model(input_ids=data_ids, attention_mask=attention_mask, token_type_ids=token_types)

                # compute the loss, move it to the cpu and convert it into a float value
                loss = criterion(output['logits'], label).detach().cpu().item()

                # add the loss to the average loss
                avg_loss += loss

            except:
                pass
    avg_loss = avg_loss / len(validation_loader)
    return avg_loss


if __name__ == '__main__':

    # properties
    data_path = 'data_mixed_labeled_preprocessed.tsv'
    model_name = 'dbmdz/bert-base-german-uncased'
    device = 'cuda:0'
    num_epochs = 10
    batch_size = 4

    # load the model
    model, tokenizer = load_model(model_name)
    model.to(device)

    # initialize the dataset
    dataset = BertDataset(data_path, tokenizer, max_length=50)

    # create train and validation loaders
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # loss function, which will be optimized
    criterion = CrossEntropyLoss()

    # optimizer which is used to optimize the model
    # optimizer = SGD(model.parameters(), lr=4e-3, momentum=0.98)
    optimizer = Adam(model.parameters(), lr=4e-5, betas=(0.9, 0.99))

    for epoch in range(num_epochs):

        print('Epoch {}'.format(epoch))

        # print('\tTraining ...')
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device)
        print(train_loss)

        # print('\tValidation...')
        # val_loss = validation_epoch(validation_loader, model, criterion, device)
        # print(val_loss)

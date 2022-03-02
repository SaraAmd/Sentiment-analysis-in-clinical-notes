import pandas as pd
import torch
import os
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def evaluate(model,dataloader, device):

    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        batch = tuple(b.to(device) for b in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids':batch[2],
        }
        with torch.no_grad():
            outputs = model(**inputs)

        targets = batch[3].to(device, dtype=torch.long)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        logits = outputs.data
        logits = logits.detach().cpu().numpy()
        label_ids = batch[3].cpu().numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    loss_avg = total_loss / len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return loss_avg, predictions, true_labels

def main():
    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    df = pd.read_csv('out1_sareh.csv')
    labels = df['label'].values

    sentiment = []
    for label in labels:
        label = label.replace( "[" , "")
        label = label.replace("]" , "")
        label = label.replace(",", "")
        sentiment.append(label)

    s = [s[:2] for s in sentiment]
    sentiments = [int(i) for i in s]




#model_name = 'Emran/ClinicalBERT_ICD10_Full'
    model_name = 'AndyJ/clinicalBERT'

    model = BertForSequenceClassification.from_pretrained(model_name)
    tokeniser = BertTokenizer.from_pretrained(model_name)

    tokens = tokeniser(df['paragraph'].tolist(), max_length= 512, padding = 'max_length', truncation = True,
                      add_special_tokens = True, return_tensors = 'pt' )

    text_IDs = tokens.input_ids
    text_mask = tokens.attention_mask
    sentiments= torch.tensor(sentiments).unsqueeze(dim=1)
    dataset = TensorDataset(text_IDs, text_mask, sentiments)
    batch_size = 16
    split = 0.9
    size = text_IDs.shape[0]
    train_size = int(size/batch_size * split)
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size , val_size])

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle= True, drop_last= True)
    val_loader = DataLoader(val_dataset, batch_size = 16, shuffle= True, drop_last= True)

    class Clinicalbert(torch.nn.Module):
        def __init__(self):
            super(Clinicalbert, self).__init__()
  #transformer
            self.hidden_states = BertForSequenceClassification.from_pretrained(model_name)
            for param in model.bert.parameters():
                param.requires_grad = False
  #clasification heads
            self.layer1 = torch.nn.Linear(768, 3)
  #prob
            self.prob = torch.nn.Softmax(dim = 1)

        def forward(self, text_IDs, attention_mask):

            embeddings = self.hidden_states(text_IDs, attention_mask)[1] #get the pooled output
            print(embeddings.shape)
            z = self.layer1(embeddings[:0]) #get the cls vector
            output= self.prob(z)
            return output



    model= Clinicalbert()
    model.to(device)

    criterian =torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(params = model.parameters() , lr = 5e-3)


    epoch = 5
    for i in range(5):
        model.train()
        for iterarion, data in (enumerate(train_loader, 0)):
            ids = data[0].to(device)
            attention = data[1].to(device)
            targets = data[2].to(device)
            out = model (ids, attention)
            loss = criterian(labels, out)
            if iterarion % 100 == 0:
                valid_loss, predictions_valid, true_valid = evaluate(model, val_loader, device)
                valid_f1 = f1_score_func(predictions_valid, true_valid)
                train_loss, predictions_train, true_train = evaluate(model, train_loader, device)
                train_f1 = f1_score_func(predictions_train, true_train)
                print("\nf1 score on valid data is: ", valid_f1)
                print("\nf1 score on train data is: ", train_f1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f'ClinicalBert_epoch{i}.model')


if __name__ == "__main__":
    main()
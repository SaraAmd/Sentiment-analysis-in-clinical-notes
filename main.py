# the code is modification of the fine tuning ber in this link https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
import pandas as pd
import torch
import torch.nn as nn
import os
from transformers import  BertConfig, BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import pdb
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random
import time


def f1_score_func(preds, labels):


    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels

    return f1_score(labels_flat, preds_flat, average='weighted')


def evaluate(model, val_dataloader, device):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    predictions, true_labels = [], []
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

        # compute f1 score
        predictions.extend(logits.detach().cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())


    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    # compute f1 score
    score = f1_score_func(predictions, true_labels)

    return val_loss, val_accuracy, score

def main():
    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    #model_name = 'Emran/ClinicalBERT_ICD10_Full'
    model_name = 'AndyJ/clinicalBERT'

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                 #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks


    batch_size = 8
    data = pd.read_csv('finalData.csv')
    data_val = pd.read_csv('finalData_val.csv')

    #specify the maximum length of our texts

    # Concatenate train data and test data
    all_texts = np.concatenate([data.text.values, data_val.text.values])

    # Encode our concatenated data
    encoded_texts = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_texts]

    # Find the maximum length
    max_len = max([len(sent) for sent in encoded_texts])
    print('Max length: ', max_len)

    # Specify `MAX_LEN`
    MAX_LEN = 512

    X_train = data.text.values
    X_val = data_val.text.values

    # Run function `preprocessing_for_bert` on the train set and the validation set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(X_train)
    val_inputs, val_masks = preprocessing_for_bert(X_val)


    #2.2. Create PyTorch DataLoader

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(data.label.values-1)
    val_labels = torch.tensor(data_val.label.values-1)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    class Clinicalbert(nn.Module):

        def __init__(self, freeze_bert=True):
            super(Clinicalbert, self).__init__()
            """
                    @param    bert: a BertModel object
                    @param    classifier: a torch.nn.Module classifier
                    @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
            """
            # Specify hidden size of BERT, hidden size of our classifier, and number of labels
            D_in, H, D_out = 768, 50, 3
            # Instantiate BERT model
            self.bert = BertModel.from_pretrained(model_name)
            #config = BertConfig.from_pretrained(model_name)

            # Instantiate an one-layer feed-forward classifier
            self.classifier = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(H, D_out)
            )
            # Freeze the BERT model
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False


        def forward(self, text_IDs, attention_mask):

            """
                    Feed input to BERT and the classifier to compute logits.
                    @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                                  max_length)
                    @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                                  information with shape (batch_size, max_length)
                    @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                                  num_labels)
            """
            # Feed input to BERT
            outputs = self.bert(input_ids=text_IDs, attention_mask=attention_mask)

            # Extract the last hidden state of the token `[CLS]` for classification task
            last_hidden_state_cls = outputs[0][:, 0, :]

            # Feed input to classifier to compute logits
            logits = self.classifier(last_hidden_state_cls)

            return logits

    def initialize_model(epochs=4):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        bert_classifier = Clinicalbert(freeze_bert=False)

        # Tell PyTorch to run the model on GPU
        bert_classifier.to(device)

        # Create the optimizer
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)
        return bert_classifier, optimizer, scheduler


    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    def set_seed(seed_value=42):
        """Set seed for reproducibility.
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device = device):
        """Train the BertClassifier model.
        """
        # Start training loop
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'F1-Score':^9} | {'Elapsed':^9}")
            print("-" * 80)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} |{time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 80)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy , F1_score= evaluate(model, val_dataloader, device)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^.4f}   | {F1_score:^9.4f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\n")

        print("Training complete!")

    set_seed(42)  # Set seed for reproducibility
    bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
    train(bert_classifier, train_dataloader, val_dataloader, epochs=2, evaluation=True)
    #torch.save(model.state_dict(), f'ClinicalBert_epoch{i}.model')


if __name__ == "__main__":
    main()
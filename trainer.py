import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
import random
import json
import argparse
import os, sys
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from data import Data

label_map = {"PAD":0, "O": 1,'B_disease':2, 'I_disease':3, 'B_crowd':4, 'I_crowd':5, 'B_body':6,'I_body':7, 'B_treatment':8, 'I_treatment':9, 'B_symptom':10,'I_symptom':11,'B_time':12,'I_time':13,'B_drug':14,'I_drug':15,'B_feature':16,'I_feature':17,'B_physiology':18,'I_physiology':19,'B_test':20,'I_test':21,'B_department':22,'I_department':23}

def train(model, iterator, optimizer, clip, grad_acc):

    model.train()
    train_start = time.time()

    optimizer.zero_grad()

    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        batch_start = time.time()
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids, masked_ids, entity_mask = batch
        # Different masking for diff epoch
        epoch_remainder = epoch % 30
        masked_ids = masked_ids[:,epoch_remainder]
        entity_mask = entity_mask[:,epoch_remainder]        

        batch_size = label_ids.shape[0]

        outputs = model(masked_ids, input_mask, labels=input_ids, output_hidden_states=True)
        loss = outputs.loss
        logits = outputs.logits
        last_hids = outputs.hidden_states[-1]
        embs = outputs.hidden_states[0]

        loss = loss / grad_acc
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
             
        if (i+1) % grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate(model, iterator):
    model.eval()
    with torch.no_grad():

        epoch_loss = 0
        correct_count = 0
        total_count = 0
        entity_correct = 0
        entity_total = 0
        flagg = 0
        for i, batch in enumerate(iterator):
            flagg = i
            batch_start = time.time()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, masked_ids, entity_mask = batch

            # Use first masking for evaluation
            masked_ids = masked_ids[:,0]
            entity_mask = entity_mask[:,0]

            batch_size = label_ids.shape[0]

            outputs = model(masked_ids, input_mask, labels=input_ids)
            loss = outputs.loss
            logits = outputs.logits

            epoch_loss += loss

            pred = torch.argmax(logits, dim=-1)
            
            match = (input_ids == pred) * input_mask
            correct_count += torch.sum(match).item()
            total_count += torch.sum(input_mask).item() 
    
            entity_match = (input_ids == pred) * entity_mask
            entity_correct += torch.sum(entity_match).item()
            entity_total += torch.sum(entity_mask).item()
    print(flagg+1)
    return epoch_loss/(flagg+1), correct_count / total_count, entity_correct / entity_total

parser = argparse.ArgumentParser()

parser.add_argument('--file_dir',  type=str, default="./data/")
parser.add_argument('--ckpt_dir', type=str, default="./")

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--bsize', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--grad_acc', type=int, default=2)
parser.add_argument('--mask_rate', type=float, default=0.7)

args = parser.parse_args()

if True:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    device="cpu"
    print("Running on ", device)

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True   # #

    FILE_DIR = args.file_dir
    CKPT_DIR = args.ckpt_dir
    BSIZE = args.bsize
    N_EPOCHS = args.n_epochs
    CLIP = args.clip
    LR = args.lr
    GRAD_ACC = args.grad_acc
    MASK_RATE = args.mask_rate
 
    ckpt_folder = '/'.join(CKPT_DIR.split('/')[:-1])
    if os.path.isdir(ckpt_folder):
        print("\nWarning! Checkpoint dir exist!.......................\n")
    else:
        os.mkdir(ckpt_folder)
        print("Checkpoints will be saved to: ", CKPT_DIR)

    print("Initializing transformer model and tokenizer...")
    model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base', return_dict=True).to(device)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=False)

    # Add entity labels as special tokens
    tokenizer.add_tokens(['<En>', '<De>', '<Es>', '<Nl>'], special_tokens=True)
    tokenizer.add_tokens(['<B-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<B-LOC>', '<I-LOC>', '<B-MISC>', '<I-MISC>','<O>'],
                         special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        # label tokens
        model.roberta.embeddings.word_embeddings.weight[-1, :] += model.roberta.embeddings.word_embeddings.weight[1810, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-2, :] += model.roberta.embeddings.word_embeddings.weight[27060, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-3, :] += model.roberta.embeddings.word_embeddings.weight[27060, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-4, :] += model.roberta.embeddings.word_embeddings.weight[31913, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-5, :] += model.roberta.embeddings.word_embeddings.weight[31913, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-6, :] += model.roberta.embeddings.word_embeddings.weight[53702, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-7, :] += model.roberta.embeddings.word_embeddings.weight[53702, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-8, :] += model.roberta.embeddings.word_embeddings.weight[3445, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-9, :] += model.roberta.embeddings.word_embeddings.weight[3445, :].clone()
        # language markers
        model.roberta.embeddings.word_embeddings.weight[-10, :] += model.roberta.embeddings.word_embeddings.weight[94854, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-11, :] += model.roberta.embeddings.word_embeddings.weight[151010, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-12, :] += model.roberta.embeddings.word_embeddings.weight[89855, :].clone()
        model.roberta.embeddings.word_embeddings.weight[-13, :] += model.roberta.embeddings.word_embeddings.weight[14941, :].clone()

    print("Loading file from: ", FILE_DIR)
    train_dataset, valid_dataset = tuple(Data(tokenizer, BSIZE, label_map, FILE_DIR, MASK_RATE).datasets)
    #print()
    train_dataloader = DataLoader(train_dataset, batch_size=BSIZE, sampler=RandomSampler(train_dataset))
    valid_dataloader = DataLoader(valid_dataset, batch_size=BSIZE)

    #test_dataloader = DataLoader(test_dataset, batch_size=BSIZE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_valid_loss = float('inf')
    best_valid_entity_acc = -float('inf')
    best_valid_entity_acc_by_acc = -float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train(model, train_dataloader, optimizer, CLIP, GRAD_ACC)
        valid_loss, valid_acc, valid_entity_acc = evaluate(model, valid_dataloader)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s',
              f'Epoch valid loss: {valid_loss:.3f} | ',
              f'Epoch valid acc: {valid_acc * 100:.2f}% | Epoch entity acc: {valid_entity_acc*100:.2f}% ')

        if valid_loss < best_valid_loss:
            print("Saving current epoch to checkpoint...")
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
            best_valid_acc = valid_acc
            best_valid_entity_acc = valid_entity_acc
            torch.save(model.state_dict(), CKPT_DIR)
        
    print("Training finished...")
    print(f'\n Best valid loss until epoch {epoch} is {best_valid_loss:.3f} at epoch {best_valid_epoch + 1}',
          f'\n valid acc is {best_valid_acc * 100:.2}%, valid entity acc is {best_valid_entity_acc * 100:.2f}%')

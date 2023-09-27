import os
import time
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from torch import load, save, argmax, sum, from_numpy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Module, Embedding, LSTM, LayerNorm, Linear, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.nn.utils import clip_grad_norm_
from utils.tokenizer import Tokenizer
from utils.pytorchtools import EarlyStopping

class CLMnet(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_dim,layer_dim,
                    batch_first=True, dropout=0.5)
        self.ln = LayerNorm(hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embeds = self.embedding(x.long())
        r_out, (h_n, c_n) = self.lstm(embeds)
        norm_out = self.ln(r_out)
        output = self.fc(norm_out)
        return output

class CLM:
    def __init__(self, config, gpu_type, pattern='pretrain'):
        assert pattern in ['pretrain', 'finetune', 'sample']
        self.tok = Tokenizer()
        self.config = config
        self.device = gpu_type
        self.pattern = pattern
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.vocab_size = len(self.tok.table)
        self.criterion = CrossEntropyLoss()

        if self.pattern == 'pretrain':
            self.init_model()
        elif self.pattern == 'finetune':
            print(f'Loading model from {self.config.pretrain_model} ...')
            self.model = load(self.config.pretrain_model, map_location=self.device)            
        else:
            print(f'Loading model from {self.config.trained_model} ...')
            self.model = load(self.config.trained_model, map_location=self.device)         
        
        self.optimizer = Adam(self.model.parameters())

    def init_model(self):
        self.model = CLMnet(self.vocab_size, self.config.embedding_dim, 
                            self.config.units, self.config.layers, self.vocab_size)

    def save_model(self, model_name):
        print('Saving model ...')
        save(self.model, model_name)
        print(f'Model saved as {os.path.basename(model_name)}')

    def pretrain(self, train_loader, valid_loader):
        sumwriter = SummaryWriter(log_dir=self.config.log_dir)
        train_loss_accu = 0
        train_loss_all = []
        train_acc_all = []
        valid_loss_accu = 0
        valid_acc_all = []
        scheduler = CosineAnnealingLR(self.optimizer, T_max=64)
        for epoch in range(self.config.num_epochs):
            since = time.time()
            print("-"*30)
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            train_loss = 0.0
            train_corrects = 0
            train_num = 0
            valid_loss = 0.0
            valid_corrects = 0
            valid_num = 0
            self.model.train()    
            for step, batch in enumerate(tqdm(train_loader, ascii=True, desc="training")):
                data_smi, target_smi = batch["input_smi"], batch["label_smi"]
                data_smi, target_smi = data_smi.to(self.device), target_smi.to(self.device)
                out = self.model(data_smi)
                out = out.reshape((-1, out.shape[-1]))
                pre_lab = argmax(out, 1)
                target = target_smi.reshape(-1)

                loss = self.criterion(out, target.long()).mean()   
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                train_loss += loss.item() * len(target_smi)
                train_loss_accu += loss.item()
                train_corrects += (sum(pre_lab == target) / data_smi.shape[1])
                train_num += len(target_smi)
                niter = epoch * len(train_loader) + step + 1
                if niter % 100 == 0:
                    sumwriter.add_scalar("train loss", train_loss_accu / niter, niter)
                    sumwriter.add_scalar("train acc", train_corrects.float().item() / train_num, niter)
            train_loss_all.append(train_loss / train_num)
            train_acc_all.append(train_corrects.float().item()/train_num)
            train_time = time.time()
            epoch_time_train = train_time - since
            print(f"Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}, Epoch Time(Train): {epoch_time_train:.2f}s")
            scheduler.step()

            self.model.eval()
            for step, batch in enumerate(tqdm(valid_loader, ascii=True, desc="validating")):
                data_smi, target_smi = batch["input_smi"], batch["label_smi"]
                data_smi, target_smi = data_smi.to(self.device), target_smi.to(self.device)
                out = self.model(data_smi)
                out = out.reshape((-1, out.shape[-1]))
                pre_lab = argmax(out, 1)
                target = target_smi.reshape(-1)

                loss = self.criterion(out, target.long()).mean()   
                valid_loss += loss.item() * len(target_smi)
                valid_loss_accu += loss.item()
                valid_corrects += (sum(pre_lab == target) / data_smi.shape[1])
                valid_num += len(target_smi)
                niter = epoch * len(valid_loader) + step + 1
                if niter % 10 == 0:
                    sumwriter.add_scalar("valid loss", valid_loss_accu / niter, niter)
                    sumwriter.add_scalar("valid acc", valid_corrects.float().item() / valid_num, niter)            
                
            avg_valid_loss = valid_loss / valid_num
            valid_acc_all.append(valid_corrects.float().item()/valid_num)
            valid_time = time.time()
            epoch_time_valid = valid_time - train_time
            print(f"Valid Loss: {avg_valid_loss:.4f} Valid Acc: {valid_acc_all[-1]:.4f}, Epoch Time (Valid): {epoch_time_valid:.2f}s")
            
            medi_model_path =  os.path.join(self.config.prj_dir, f'CLM-pretrain-{epoch+1}-{avg_valid_loss:.4f}.pkl')
            self.early_stopping.path = medi_model_path
            self.early_stopping(avg_valid_loss, self.model) # save the model when validation loss decrease
            if self.early_stopping.early_stop:
                print("Early stopping!")
                break
        best_model_path = self.early_stopping.saved_paths[-1]
        best_epoch = int(os.path.basename(best_model_path).split("-")[2])
        best_valid_loss = float(os.path.basename(best_model_path).split("-")[3].rstrip(".pkl"))
        best_model = load(best_model_path)
        self.config.pretrain_model = os.path.join(self.config.prj_dir, f'CLM-pretrain_best-{best_epoch}-{best_valid_loss:.4f}.pkl')
        save(best_model, self.config.pretrain_model) # save the model with the best validation performance
    
    def finetune(self, train_loader, valid_loader):
        sumwriter = SummaryWriter(log_dir=self.config.log_dir)
        train_loss_accu = 0
        train_loss_all = []
        train_acc_all = []
        valid_loss_accu = 0
        valid_acc_all = []
        scheduler = CosineAnnealingLR(self.optimizer, T_max=32)
        for epoch in range(self.config.num_epochs):
            since = time.time()
            print("-"*30)
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            train_loss = 0.0
            train_corrects = 0
            train_num = 0
            valid_loss = 0.0
            valid_corrects = 0
            valid_num = 0            
            self.model.train()
            for step, batch in enumerate(tqdm(train_loader, ascii=True, desc="training")):
                data_smi, target_smi = batch["input_smi"], batch["label_smi"]
                data_smi, target_smi = data_smi.to(self.device), target_smi.to(self.device)
                out = self.model(data_smi)
                out = out.reshape((-1, out.shape[-1]))
                pre_lab = argmax(out, 1)
                target = target_smi.reshape(-1)
                # freeze embedding layer when finetuning
                for param in self.model.embedding.parameters():
                    param.requires_grad = False                
                loss = self.criterion(out, target.long()).mean()   
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                train_loss += loss.item() * len(target_smi)
                train_loss_accu += loss.item()
                train_corrects += (sum(pre_lab == target) / data_smi.shape[1])
                train_num += len(target_smi)
                niter = epoch * len(train_loader) + step + 1
                if niter % 100 == 0:
                    sumwriter.add_scalar("train loss", train_loss_accu / niter, niter)
                    sumwriter.add_scalar("train acc", train_corrects.float().item() / train_num, niter)
            train_loss_all.append(train_loss / train_num)
            train_acc_all.append(train_corrects.float().item()/train_num)
            train_time = time.time()
            epoch_time_train = train_time - since
            print(f"Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}, Epoch Time (Train): {epoch_time_train:.2f}s")
            scheduler.step()

            self.model.eval()
            for step, batch in enumerate(tqdm(valid_loader, ascii=True, desc="validating")):
                data_smi, target_smi = batch["input_smi"], batch["label_smi"]
                data_smi, target_smi = data_smi.to(self.device), target_smi.to(self.device)
                out = self.model(data_smi)
                out = out.reshape((-1, out.shape[-1]))
                pre_lab = argmax(out, 1)
                target = target_smi.reshape(-1)

                loss = self.criterion(out, target.long()).mean()   
                valid_loss += loss.item() * len(target_smi)
                valid_loss_accu += loss.item()
                valid_corrects += (sum(pre_lab == target) / data_smi.shape[1])
                valid_num += len(target_smi)
                niter = epoch * len(valid_loader) + step + 1
                if niter % 10 == 0:
                    sumwriter.add_scalar("valid loss", valid_loss_accu / niter, niter)
                    sumwriter.add_scalar("valid acc", valid_corrects.float().item() / valid_num, niter)            
                
            avg_valid_loss = valid_loss / valid_num
            valid_acc_all.append(valid_corrects.float().item()/valid_num)
            valid_time = time.time()
            epoch_time_valid = valid_time - train_time
            print(f"Valid Loss: {avg_valid_loss:.4f} Valid Acc: {valid_acc_all[-1]:.4f}, Epoch Time (Valid): {epoch_time_valid:.2f}s")
            
            medi_model_path =  os.path.join(self.config.prj_dir, f'CLM-finetune-{epoch+1}-{avg_valid_loss:.4f}.pkl')
            self.early_stopping.path = medi_model_path
            self.early_stopping(avg_valid_loss, self.model) # save the model when validation loss decrease
            if self.early_stopping.early_stop:
                print("Early stopping!")
                break
        best_model_path = self.early_stopping.saved_paths[-1]
        best_epoch = int(os.path.basename(best_model_path).split("-")[2])
        best_valid_loss = float(os.path.basename(best_model_path).split("-")[3].rstrip(".pkl"))
        best_model = load(best_model_path)
        self.config.finetune_model = os.path.join(self.config.prj_dir, f'CLM-finetune_best-{best_epoch}-{best_valid_loss:.4f}.pkl')
        save(best_model, self.config.finetune_model) # save the model with the best validation performance

    def sample(self, start='G'):
        num = self.config.sampling_num
        sampled = []
        for _ in tqdm(range(num)):
            sampled.append(self.sample_seq(start))
        return sampled

    def sample_seq(self, sequence):
        while (sequence[-1] != 'E') and (len(self.tok.tokenize(sequence)) <=
                                         self.config.tokens_max_len):
            x = self.tok.to_idxes(self.tok.tokenize(sequence))#x.shape(1,num_tokens)
            x = from_numpy(x).float().to(self.device)
            preds = softmax(self.model(x)[0][-1], dim=0).detach().cpu().numpy()
            next_idx = self.sample_with_temp(preds)
            sequence += self.tok.table[next_idx]
        
        sequence = sequence[1:].rstrip('E')
        return sequence

    def sample_with_temp(self, preds):
        streched = np.log(preds) / self.config.sampling_temp
        streched_probs = np.exp(streched) / np.sum(np.exp(streched))
        return np.random.choice(range(len(streched)), p=streched_probs)
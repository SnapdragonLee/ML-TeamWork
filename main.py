from tqdm import tqdm
import pandas as pd
import os
from functools import partial
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel, AutoConfig, RobertaTokenizer, \
    RobertaModel
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#################################################################
# 训练数据调整 145行 max_len设为256 更高则设为500 (BERT最大限制为512)  #
#################################################################

max_len = 128 # 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
train = pd.read_csv('DataSet/Train.csv', index_col=False)

data = list()
for line in tqdm(train.iterrows()):
    data.append(line)

train.columns = ['num', 'text', 'emotions', 'id']
train['positive'] = str(0)
train['negative'] = str(0)
train['neutral'] = str(0)

test = pd.read_csv('DataSet/Test.csv', index_col=False)
train = train[1:-1]

# 数据处理

for i in range(1, 36205):
    if train.loc[i, 'emotions'] == "positive":
        train.loc[i, 'positive'] = 1
    elif train.loc[i, 'emotions'] == "negative":
        train.loc[i, 'negative'] = 1
    else:
        train.loc[i, 'neutral'] = 1
    i = i + 1

test.columns = ['num', 'text', 'id']
test['positive'] = str(0)
test['negative'] = str(0)
test['neutral'] = str(0)

train.to_csv('DataSet/train_n.csv',
             columns=['num', 'text', 'emotions', 'id', 'positive', 'negative', 'neutral'],
             sep='\t',
             index=False)

test.to_csv('DataSet/test_n.csv',
            columns=['num', 'text', 'id', 'positive', 'negative', 'neutral'],
            sep='\t',
            index=False)

# 定义dataset
target_cols = ['positive', 'negative', 'neutral']


class RoleDataset(Dataset):
    def __init__(self, tokenizer, max_len, mode='train'):
        super(RoleDataset, self).__init__()
        if mode == 'train':
            self.data = pd.read_csv('DataSet/train_n.csv', sep='\t')
            # print(_train_dataframe['text'])
        else:
            self.data = pd.read_csv('DataSet/test_n.csv', sep='\t')
        self.texts = self.data['text'].tolist()
        self.labels = self.data[target_cols].to_dict('records')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(text,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              return_token_type_ids=True,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors='pt', )

        sample = {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        for label_col in target_cols:
            sample[label_col] = torch.tensor(label[label_col] / 3.0, dtype=torch.float)
        return sample

    def __len__(self):
        return len(self.texts)


# create dataloader
def create_dataloader(dataset, batch_size, mode='train'):
    shuffle = True if mode == 'train' else False

    if mode == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# 加载预训练模型
# roberta


#PRE_TRAINED_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment'
#PRE_TRAINED_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment'
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
#tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
base_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)  # 加载预训练模型


# model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)

# 模型构建
def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return


class MLModelLite(nn.Module):
    def __init__(self, n_classes, model_name):
        super(MLModelLite, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.base = BertModel.from_pretrained(model_name, config=config)

        dim = 1024 if 'large' in model_name else 768

        self.attention = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        # self.attention = AttentionHead(h_size=dim, hidden_dim=512, w_drop=0.0, v_drop=0.0)

        self.out_pos = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_neg = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_neu = nn.Sequential(
            nn.Linear(dim, n_classes)
        )

        init_params([self.out_pos, self.out_neg, self.out_neu])

    def forward(self, input_ids, attention_mask):
        roberta_output = self.base(input_ids=input_ids,
                                   attention_mask=attention_mask)

        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        # print(weights.size())
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        # context_vector = weights

        pos = self.out_pos(context_vector)
        neg = self.out_neg(context_vector)
        neu = self.out_neu(context_vector)

        return {
            'positive': pos, 'negative': neg, 'neutral': neu
        }


# 参数配置
EPOCHS = 2
weight_decay = 0.0
data_path = 'data'
warmup_proportion = 0.0
batch_size = 4 # 16
lr = 1e-5

warm_up_ratio = 0.000

trainset = RoleDataset(tokenizer, max_len, mode='train')
train_loader = create_dataloader(trainset, batch_size, mode='train')
valset = RoleDataset(tokenizer, max_len, mode='test')
valid_loader = create_dataloader(valset, batch_size, mode='test')

model = MLModelLite(n_classes=1, model_name=PRE_TRAINED_MODEL_NAME)

model.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # correct_bias=False,
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warm_up_ratio * total_steps,
    num_training_steps=total_steps
)

criterion = nn.BCEWithLogitsLoss().to(device)


def do_train(model, criterion, optimizer, scheduler):
    model.train()
    global_step = 0
    tic_train = time.time()
    log_steps = 100
    for epoch in range(EPOCHS):
        losses = []
        for step, sample in enumerate(train_loader):
            # input_ids = sample["input_ids"].cuda()
            input_ids = sample["input_ids"].to(device)
            # attention_mask = sample["attention_mask"].cuda()
            attention_mask = sample["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # loss_pos = criterion(outputs['positive'], sample['positive'].view(-1, 1).cuda())
            # loss_neg = criterion(outputs['negative'], sample['negative'].view(-1, 1).cuda())
            # loss_neu = criterion(outputs['neutral'], sample['neutral'].view(-1, 1).cuda())
            loss_pos = criterion(outputs['positive'], sample['positive'].view(-1, 1).to(device))
            loss_neg = criterion(outputs['negative'], sample['negative'].view(-1, 1).to(device))
            loss_neu = criterion(outputs['neutral'], sample['neutral'].view(-1, 1).to(device))

            loss = loss_pos + loss_neg + loss_neu

            losses.append(loss.item())

            loss.backward()

            #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train),
                         float(scheduler.get_last_lr()[0])))


do_train(model, criterion, optimizer, scheduler)

"""
##############################################
# 这部分代码为模型软投票
##############################################
def merge_pred(models: list, valid_loaders: list):
    label_preds = np.zeros(6)
    for model_num, model in enumerate(models):
        valid_loader = valid_loaders[model_num]
        test_pred = predict(model, valid_loader)
        single_label_preds = []
        for col in target_cols:
            preds = test_pred[col]
            single_label_preds.append(preds)
        label_preds = label_preds + np.stack(single_label_preds, axis=1)
    label_preds = label_preds / len(models)
    submit = pd.read_csv(path.get_dataset_path('submit_example.tsv'), sep='\t')
    print(len(label_preds[0]))
    sub = submit.copy()
    sub['emotion'] = label_preds.tolist()
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
    sub.to_csv(
        path.build_submit_path('MergeAns.tsv'),
        sep='\t',
        index=False)
    sub.head()


models = list()
valid_loaders = list()
for i, model_name in enumerate(path.MODELS_NAME):
    models.append(torch.load(path.build_model_path(model_name)))
    val_set = RoleDataset(tokenizer, max_len, mode='test', test_num=i)
    valid_loader = create_dataloader(val_set, batch_size, mode='test')
    valid_loaders.append(valid_loader)
merge_pred(models, valid_loaders)
##############################################
"""


# 模型预测

def predict(model, test_loader):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    # model.cuda()
    model.to(device)
    for batch in tqdm(test_loader):
        # b_input_ids = batch['input_ids'].cuda()
        b_input_ids = batch['input_ids'].to(device)
        # attention_mask = batch["attention_mask"].cuda()
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
            for col in target_cols:
                out2 = logists[col].sigmoid().squeeze(1) * 3.0
                test_pred[col].extend(out2.cpu().numpy().tolist())

    return test_pred


model.eval()

test_pred = defaultdict(list)
for step, batch in tqdm(enumerate(valid_loader), desc='Final Pred'):
    b_input_ids = batch['input_ids'].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
        for col in target_cols:
            out2 = logists[col].sigmoid().squeeze(1) * 3.0
            test_pred[col].append(out2.cpu().numpy())

    print(test_pred)
    break

test_pred = predict(model, valid_loader)

print(test_pred['positive'][:10])
print(len(test_pred['positive']))

label_preds = []

for col in target_cols:
    preds = test_pred[col]
    label_preds.append(preds)
print(len(label_preds[0]))

with open('./submission.txt', 'w') as answer:
    for ans in tqdm(range(0, len(test_pred['positive']))):
        if label_preds[0][ans] >= label_preds[1][ans]:
            if label_preds[0][ans] < label_preds[2][ans]:
                target = 2
            if label_preds[0][ans] > label_preds[2][ans]:
                target = 0
        elif label_preds[0][ans] < label_preds[1][ans]:
            if label_preds[1][ans] < label_preds[2][ans]:
                target = 2
            if label_preds[1][ans] > label_preds[2][ans]:
                target = 1
        answer.write(target_cols[target] + "\n")

    '''
    sub = submit.copy()
    sub['emotion'] = np.stack(label_preds, axis=1).tolist()
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
    sub.to_csv(
        path.build_submit_path('K test link{0} {1}.tsv').format(LINK_NUM, PRE_TRAINED_MODEL_NAME.split('/')[-1]),
        sep='\t',
        index=False)
    sub.head()
    '''

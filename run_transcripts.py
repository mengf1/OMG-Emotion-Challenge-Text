import util
import numpy as np
import numpy
import model
import argparse
import torch
import os
import datetime
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch import nn


torch.cuda.set_device(2)

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
args = parser.parse_args()

# data set - train
input_file = "../omg_TrainTranscripts.csv"
input_file2 = "../omg_TrainVideos.csv"
data = util.process_transcripts(input_file, input_file2)
# prepare data
X, Y, corpus = util.read_data_omg(data)

# data set - dev
input_file = "../omg_ValidationTranscripts.csv"
input_file2 = "../omg_ValidationVideos.csv"
data = util.process_transcripts(input_file, input_file2)
# prepare data
Xv, Yv, corpus = util.read_data_omg(data)

# prepare embedding
emb_file = "../glove/glove.6B.100d.txt"
w2v = util.load_embeddings_from_glove(emb_file, corpus.word2index)

# model
emb_dim = 100
lstm_model = model.LSTM_Text(emb_dim, 128, 32).cuda()

def sent_to_tensor(x):
    tensor = torch.zeros(len(x), 1, emb_dim)
    for index, tok_id in enumerate(x):
        tensor[index][0] = torch.from_numpy(w2v[tok_id])
    return tensor

def eval(X, Y, model):
    model.cpu()
    y_pred = [[],[]]
    for i in range(len(X)):
        x = X[i]
        if len(x) == 0:
            y_pred[0].append(0.)
            y_pred[1].append(0.)
        else:
            sent_tensor = Variable(sent_to_tensor(x))
            hidden = model.init_hidden()
            cell = model.init_hidden()
            for i in range(len(x)):
                logit1, logit2, hidden, cell = model(sent_tensor[i], hidden, cell)
            y_pred[0].append(logit1.data.numpy()[0])
            y_pred[1].append(logit2.data.numpy()[0])
    y1 = np.array(y_pred[0])
    y_t1 = [Y[i][0] for i in range(len(Y))]
    y_t1 = np.array(y_t1)
    ccc1, _ = ccc(y_t1, y1)
    mse1 = mse(y_t1, y1)
    y2 = np.array(y_pred[1])
    y_t2 = [Y[i][1] for i in range(len(Y))]
    y_t2 = np.array(y_t2)
    ccc2, _ = ccc(y_t2, y2)
    mse2 = mse(y_t2, y2)
    model.cuda()
    return ccc1, ccc2, mse1, mse2

# from OMG
def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true,y_pred)

# from OMG
def ccc(y_true, y_pred):
    #print(y_true, y_pred)
    true_mean = numpy.mean(y_true)
    true_variance = numpy.var(y_true)
    pred_mean = numpy.mean(y_pred)
    pred_variance = numpy.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = numpy.std(y_pred)

    std_gt = numpy.std(y_true)


    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc, rho

# start to train
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=args.lr)

train_res = []
dev_res = []
criterion = nn.MSELoss()
Max_Iter = 20
for epoch in range(Max_Iter): 
    for i in range(len(X)):
        if len(X[i]) == 0:
            continue
        else:
            line_tensor = Variable(sent_to_tensor(X[i])).cuda()
            label1 = Y[i][0]
            label1 = Variable(torch.FloatTensor([label1])).cuda()
            label2 = Y[i][1]
            label2 = Variable(torch.FloatTensor([label2])).cuda()

            # lstm
            hidden = lstm_model.init_hidden().cuda()
            cell = lstm_model.init_hidden().cuda()

            for j in range(len(X[i])):
                logit1, logit2, hidden, cell = lstm_model(line_tensor[j], hidden, cell)
            
            optimizer.zero_grad()
            loss1 = criterion(logit1, label1)
            loss2 = criterion(logit2, label2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
           
    ccc1, ccc2, mse1, mse2 = eval(X, Y, lstm_model)
    print("Train", ccc1, ccc2, mse1, mse2)
    train_res.append((ccc1, ccc2, mse1, mse2))
    ccc1, ccc2, mse1, mse2= eval(Xv,Yv, lstm_model)
    print("Dev", ccc1, ccc2, mse1, mse2)
    dev_res.append((ccc1, ccc2, mse1, mse2))
    print("Epoch[Finished]", epoch)
print("Train", train_res)
print("Dev", dev_res)

    


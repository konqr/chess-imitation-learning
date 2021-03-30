# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:02:03 2021

@author: konqr
"""
import math
import random
import pickle
import torch 
import torch.nn as nn
from torch import optim
import numpy as np
from functools import reduce
# import os
from collections import defaultdict as ddict
import chess.pgn
import chess.engine
import chessenv
import prioritized_memory

ENGINE = chess.engine.SimpleEngine.popen_uci('sf.exe')

class ChessModel(nn.Module):
    
    def __init__(self):
        super(ChessModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(8,32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1))
        self.fc_value = nn.Linear(1536, 1)
        self.fc_actions = nn.Linear(1536, len(chessenv.ChessEnv().action_space()))
        
    def forward(self, s):
        x = chessenv.state_from_board(s)
        # s.push(a)
        # x2 = chessenv.state_from_board(s)
        x = torch.Tensor([x]) # + x2])
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out)
        val = self.fc_value(out)
        act = self.fc_actions(out)
        out = act.add(val-act.mean(dim=-1).unsqueeze(-1))
        return out
        
def loaddata(path='data/Fischer.pgn'):
    pgn = open(path)
    game_list = []
    while True:
        g = chess.pgn.read_game(pgn)
        if not g:
            break
        else:
            game_list.append(g)
    return game_list

def transformsarsa(game,color,pretendor,episodeId):
    try:
        with open ('data/'+pretendor+str(episodeId), 'rb') as fp:
            results = pickle.load(fp)
    except:
        s = chess.Board()
        moves = list(game.mainline_moves())
        results = []
        isWhite = True #corresponds to White
        for move in moves:
            s_ = s.copy()
            #info = ENGINE.analyse(s_, chess.engine.Limit(depth=20))
            #r1 = info['score'].relative.score(mate_score=1e5)
            s_.push(move)
            info = ENGINE.analyse(s_, chess.engine.Limit(depth=20))
            r2 = info['score'].relative.score(mate_score=1e5)
            done = s_.is_game_over()
            if ((color == 'White') and isWhite) or ((color=='Black')and not isWhite):
                results.append((s,move,r2,s_,done))
            s = s_.copy()
            isWhite = not isWhite    
        with open('data/'+pretendor+str(episodeId), 'wb') as fp:
            pickle.dump(results, fp)
    return results

def legal_action(board):
    return list(board.legal_moves)
    
make_matrix = chessenv.make_matrix

class ValueCalculator:
    def __init__(self, Net, actionFinder):
        self.predictNet = Net()
        self.targetNet = Net()
        self.actionFinder = actionFinder
        self.updateTargetNet()
        # (state => Action :List[List])

    def calcQ(self, net, s, A):
        if isinstance(s,tuple):
            result = []
            for s_i, A_i in zip(s,A):
                res = net(s_i)
                if not isinstance(A_i,str): A_i = A_i.uci()
                index_A = chessenv.ChessEnv().action_space().index(A_i)
                result.append(res[index_A].item())
            return torch.tensor(result)
        return net(s)

    def sortedA(self, state):
        # return sorted action
        net = self.predictNet
        net.eval()
        A = [x.uci() for x in self.actionFinder(state)]
        A_all = chessenv.ChessEnv().action_space()
        Q = self.calcQ(net, state, A)
        A_sorted = [a for q,a in sorted(zip(Q, A_all),reverse=True,key=lambda x:x[0]) if a in A]
        net.train()
        return A_sorted

    def updateTargetNet(self):
        self.targetNet.load_state_dict(self.predictNet.state_dict())
        self.targetNet.eval()

class DeepImitationLearning:
    def  __init__(self,Net,eps=0.9,lr=1e-3,gamma=0.9,minibatchsize=25,memsize=500,freq_target_replace=100, lambda1=1.0, lambda2 = 1.0, lambda3=1e-5, n_step = 1):
        self.eps = eps
        self.gamma = gamma
        self.minibatchsize = minibatchsize
        self.memsize= memsize
        self.freq_target_replace = freq_target_replace
        self.target_replace_counter = 0
        self.lambda1 = lambda1 #n-step return
        self.lambda2 = lambda2 #supervised loss
        self.lambda3 = lambda3 #regularization
        self.n_step = n_step
        self.replay = prioritized_memory.Memory(capacity = self.memsize)
        self.loss = prioritized_memory.WeightedMSE()
        self.value_calc = ValueCalculator(Net, legal_action) #make sure to make actionFinder = legalactions only in chess.board()
        self.opt = optim.Adam(self.value_calc.predictNet.parameters(),lr=lr,weight_decay = lambda3)
        self.margin = 0.8 #what?
        self.demoReplay = ddict(list)
    
    def act(self,state):
        # state = torch.Tensor(state)
        A = self.value_calc.sortedA(state)
        r = random.random()
        a = A[0] if self.eps>r else random.sample(A,1)[0]
        return a
    
    def sample(self):
        return self.replay.sample(self.minibatchsize)
    
    def store(self,data):
        self.replay.add(data)
        
    def storeDemoTransition(self,s,a,r,s_,done,demoEpisode):
        # s = torch.Tensor(s)
        # s_ = torch.Tensor(s_)
        episodeReplay = self.demoReplay[demoEpisode]
        index = len(episodeReplay)
        data = (s,a,r,s_,done,(demoEpisode,index))
        self.store(data)
        
    def storeTransition(self,s,a,r,s_,done):
        # s = torch.Tensor(s)
        # s_ = torch.Tensor(s_)
        self.store((s,a,r,s_,done,None))
        
    def calcTD(self, samples):
        #self.value_calc.predictNet.sample()
        all_s, all_a, all_r, all_s_, all_done, *_= zip(*samples)
        maxA = [self.value_calc.sortedA(s_)[0] for s_ in all_s_]
        Qtarget = torch.Tensor(all_r)
        Qtarget[torch.tensor(all_done) != 1] += self.gamma*self.value_calc.calcQ(self.value_calc.targetNet, all_s_, maxA)[torch.tensor(all_done) != 1]
        Qpredict = self.value_calc.calcQ(self.value_calc.predictNet,all_s,all_a)
        return Qpredict, Qtarget
    
    def JE(self, samples):
        loss= torch.tensor(0.0)
        count= 0
        for s, aE, *_, isdemo in samples:
            if isdemo is None:
                continue
            A = self.value_calc.sortedA(s)
            if len(A) == 1:
                continue
            QE = self.value_calc.calcQ(self.value_calc.predictNet,s,aE)
            A1, A2 = np.array(A)[:2]
            maxA = A2 if (A1==aE) else A1
            Q = self.value_calc.calcQ(self.value_calc.predictNet,s,maxA)
            loss_i = (Q-QE)[(Q+self.margin) > QE]
            loss += sum(loss_i)
            count += len(loss_i)
        return loss/count if count!=0 else loss
    
    def Jn(self, samples, Qpredict):
        loss = torch.tensor(0.0)
        count = 0
        for i, (s,a,r,s_,done,isdemo) in enumerate(samples):
            if isdemo is None:
                continue
            episode, idx = isdemo
            nidx = idx + self.n_step
            lepoch = len(self.demoReplay[episode])
            if nidx > lepoch:
                continue
            count+=1
            ns,na,nr,ns_,ndone,_ = zip(*self.demoReplay[episode][idx:nidx])
            ns,na,ns_,ndone = ns[-1], na[-1], ns_[-1], ndone[-1]
            discountedR = reduce(lambda x,y: (x[0]+self.gamma**x[1]*y,x[1]+1),nr,(0,0))[0]
            maxA = self.value_calc.sortedA(ns_)[0]
            target = discountedR if ndone else discountedR + self.gamma**self.n_step*self.value_calc.calcQ(self.value_calc.targetNet, ns_, maxA)
            predict = Qpredict[i]
            loss+= (target - predict)**2
        return loss/count
    
    def update(self):
        self.opt.zero_grad()
        samples,idxs, IS = self.sample()
        Qpredict, Qtarget = self.calcTD(samples)
        for i in range(self.minibatchsize):
            error = math.fabs(float(Qpredict[i]-Qtarget[i]))
            self.replay.update(idxs[i], error)
        Jtd = self.loss(Qpredict, Qtarget, IS*0+1)
        print(Jtd)
        JE = self.JE(samples)
        print(JE)
        Jn = self.Jn(samples,Qpredict)
        print(Jn)
        J = Jtd + self.lambda2*JE + self.lambda1*Jn
        print(J)
        J.backward()
        self.opt.step()
        if self.target_replace_counter >= self.freq_target_replace:
            self.target_replace_counter = 0
            self.value_calc.updateTargetNet()
        else:
            self.target_replace_counter+=1
     
import shutil
def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir +'/checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_fpath)

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    env = chessenv.ChessEnv()
    s = env.reset()
    dqn = DeepImitationLearning(ChessModel,minibatchsize=128)
    epochs = 100
    eps_start = 0.05
    eps_end = 0.95
    N = 1 - eps_start
    lam = - math.log((1-eps_end)/N)/epochs
    total = 0
    count = 0
    start = 0
    process = []
    pretendor = 'Fischer'
    
    game_list = loaddata('data/'+pretendor+'.pgn')
    episodeId=0
    for game in game_list:
        demoEpisode = game.headers
        color_player = 'White' if pretendor in demoEpisode['White'] else 'Black'
        sarsa_game = transformsarsa(game,color_player,pretendor,episodeId)
        for s,a,r,s_,done in sarsa_game:
            start+=1
            dqn.storeDemoTransition(s, a, r, s_, done, episodeId)
        print("Loaded game : ",demoEpisode)
        episodeId +=1
            
    dqn.replay.tree.start = start
    for i in range(500):
        if i %1 == 0:
            print('pretrain:',i)
        dqn.update()
        # checkpoint = {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }
        # save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
        
        
    for i in range(epochs):
        print(i)
        dqn.eps = 1 - N*math.exp(-lam*i)
        count += 1
        while True:
            a= dqn.act(s)
            s_,r,done,_ = env.step(a[0])
            total += r
            dqn.storeTransition(s, a, r, s_, done)
            dqn.update()
            s = s_.copy()
            if done or total >= 600:
                s=env.reset()
                print('total:',total)
                process.append(total)
                break
    
    
# TODO: change the reward sign for black player






# def load_ckp(checkpoint_fpath, model, optimizer):
#     checkpoint = torch.load(checkpoint_fpath)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     return model, optimizer, checkpoint['epoch']

# model = MyModel(*args, **kwargs)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# ckp_path = "path/to/checkpoint/checkpoint.pt"
# model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
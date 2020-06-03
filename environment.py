import torch
import numpy as np
import time
class  Environment:
    def __init__(self,net,dataset,state_dim=1024,action_dim=3,length=32):
        self.baseNetwork = net
        self.dataset = dataset
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.length = length
        self._max_steps = 50
        self.net = net
        self.dataset = dataset
        # buffer for p_{t-1}
        self.accPool = {}
        # buffer for predicted lab
        self.labPool = {}
        # buffer for indices
        self.indicesPool = np.zeros([len(dataset),length])
        # buffer for count
        self.countPool = {}
        self.init_pool()

    def init_pool(self):
        # # save
        # start = time.time()
        # for i in range(len(self.dataset)):
        #     print(f"{i} / {len(self.dataset)}")
        #     data = self.dataset[i]
        #     self.indicesPool[i] = self.get_sample_indices(len(data),mode='uniform')
        #     self.countPool[i] = 0
        # end = time.time()
        # print(f"Init indices pool cost {(end-start):.2f} s")
        # np.save('cache/CSL_isolated_indices.npy',self.indicesPool)
        # load
        self.indicesPool = np.load('cache/CSL_isolated_indices.npy')
        for i in range(len(self.dataset)):
            self.countPool[i] = 0

    def step(self,action,videoInd):
        data, lab = self.preprocess_data(self.dataset[videoInd])
        # execute action
        lastIndices = self.indicesPool[videoInd]
        curIndices = self.move(lastIndices,action,len(data))
        self.indicesPool[videoInd] = curIndices
        # forward
        data = data[:,curIndices,:,:]
        with torch.no_grad():
            feature = self.net.get_feature(data)
            h = self.net.classify(feature)[0]
        # get next state
        next_state = feature.cpu().data.numpy()[0]
        # assign reward
        p_t = h[lab]
        predict = h.argmax(-1)
        reward = 0
        if videoInd in self.accPool.keys():
            p_last = self.accPool[videoInd]
            if p_t > p_last: reward = 1
            elif p_t == p_last: reward = 0
            else: reward = -1
            self.accPool[videoInd] = p_t
        else:
            self.accPool[videoInd] = p_t
        if videoInd in self.labPool.keys():
            last_predict = self.labPool[videoInd]
            if predict.item()==lab and last_predict!=lab: reward = 50
            elif predict.item()!=lab and last_predict==lab: reward = -50
            self.labPool[videoInd] = predict.item()
        else:
            self.labPool[videoInd] = predict.item()
        # decide if done
        done = 0
        self.countPool[videoInd] += 1
        if self.countPool[videoInd] > self._max_steps: done = 1     # scenario 1: exceed max_steps
        #if (action.argmax(-1)==1): done = 1                        # scenario 2: all stop  
        return next_state, reward, done

    def move(self,lastIndices,action,num_frames):
        movements = action.argmax(-1)
        curIndices = lastIndices + movements
        curIndices = np.clip(curIndices,0,num_frames-1)
        curIndices.sort()
        return curIndices

    def preprocess_data(self,batch):
        data, lab = batch
        data = torch.Tensor(data).unsqueeze(0).cuda()
        return data, lab

    def get_sample_indices(self,num_frames,mode='random'):
        # the first frame has been dropped when transfer to npy
        indices = np.linspace(0,num_frames-1,self.length).astype(int)
        if mode=='random':
            interval = (num_frames-1)//self.length
            if interval>0:
                jitter = np.random.randint(0,interval,self.length)
            else:
                jitter = 0
            jitter = (np.random.rand(self.length)*interval).astype(int)
            indices = np.sort(indices+jitter)
        indices = np.clip(indices,0,num_frames-1)
        return indices

    def seed(self,seed):
        np.random.seed(seed)

    def reset(self,videoInd):
        if videoInd in self.accPool.keys():
            self.accPool.pop(videoInd)
        if videoInd in self.labPool.keys():
            self.labPool.pop(videoInd)
        data, lab = self.preprocess_data(self.dataset[videoInd])
        initIndices = self.get_sample_indices(len(data),mode='uniform')
        self.indicesPool[videoInd] = initIndices
        self.countPool[videoInd] = 0
        # forward
        data = data[:,initIndices,:,:]
        with torch.no_grad():
            feature = self.net.get_feature(data)
        state = feature.cpu().data.numpy()[0]
        return state

    def sample(self):
        action = np.random.randint(0,self.action_dim,size=self.length)
        action = np.array([[1 if i == l else 0 for i in range(self.action_dim)] for l in action])
        return action
        

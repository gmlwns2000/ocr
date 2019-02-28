#!/usr/bin/env python
# coding: utf-8

# # COCO Text Dataset Reader
# Written by Heejun Lee. Latium Project 2019

# In[2]:


import sys
import time
import coco_text
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import display
import _pickle as cPickle
def in_notebook():
    defined = True
    try:
        get_ipython
    except NameError:
        defined = False
    return ('ipykernel' in sys.modules) and defined
if in_notebook():
    get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def rectIsIn(parent, child):
    pivotRect = child
    roiRect = parent
    return pivotRect[0] >= roiRect[0] and pivotRect[1] >= roiRect[1] and         pivotRect[0]+pivotRect[2] <= roiRect[0]+roiRect[2] and pivotRect[1]+pivotRect[3] <= roiRect[1]+roiRect[3]

def rectClamp(parent, child):
    return [max(child[0], 0), max(child[1], 0),             min(child[2], parent[0]+parent[2]-child[0]), min(child[3], parent[1]+parent[3]-child[1])]


# In[7]:


class coco_text_dataset:
    def __init__(self, jsonpath='cocotext.v2.json',
                 imgdir=r'F:\Library\coco\images\train2014',
                 areaMax=1000000, areaMin = 14*14, 
                 batchW=416, batchH=416,
                 outputGridW=14, outputGridH=14,
                 catIds=[('legibility','legible')],
                 minSize=[0,14],
                 textDim=256, styleDim=32, latentDim=256,
                 bboxCount=1, ignoreCache=False):
        s = time.time()
        self.imgDir = imgdir
        print("loading data...", time.time()-s)
        self.data = coco_text.COCO_Text(jsonpath, ignoreCache)
        print('getAnnos...', time.time()-s)
        self.anno = self.data.getAnnIds(imgIds=self.data.train, 
                                        catIds=catIds, 
                                        areaRng=[areaMin,areaMax],
                                        minSize=minSize)
        print("loaded!", time.time()-s)
        self.batchW = batchW
        self.batchH = batchH
        self.outputGridW = outputGridW
        self.outputGridH = outputGridH
        self.areaMax = areaMax
        self.areaMin = areaMin
        self.bboxCount = bboxCount
        self.minSize = minSize
        self.catIds = catIds
        self.textDim = textDim
        self.styleDim = styleDim
        self.latentDim = latentDim
    
    def getFrame(self, errorStack = []):
        if self.bboxCount > 1:
            raise Exception('too much bbox!')
        
        #Return image dim format: HWC, Return code dim format: CHW
        frame = {'img':None,'bbox':None,'text':None,'style':None,'latent':None}
        #Get pivot
        pivot = self.data.loadAnns(self.anno[random.randint(0, len(self.anno)-1)])[0]
        img = self.data.loadImgs(pivot['image_id'])[0]
        imgRect = [0,0,img['width'], img['height']]
        pivotRect = rectClamp(imgRect, pivot['bbox'])

        #Find ROI
        roiWidth = max(pivotRect[2]+4, self.batchW)
        roiHeight = max(pivotRect[3]+4, self.batchH)
        roiRect = [
            pivotRect[0]-(roiWidth - pivotRect[2])*random.random(),
            pivotRect[1]-(roiHeight - pivotRect[3])*random.random(),
            roiWidth, roiHeight
        ]
        roiRect[0] = min(max(roiRect[0],0),max(0,img['width']-roiWidth))
        roiRect[1] = min(max(roiRect[1],0),max(0,img['height']-roiHeight))
        roiRect = rectClamp(imgRect, roiRect)
        
        #Get annotation in ROI
        annos = self.data.getAnnIds(imgIds=[img['id']], 
                                    catIds=self.catIds,
                                    areaRng=[self.areaMin, self.areaMax],
                                    minSize=self.minSize)
        annos = self.data.loadAnns(annos)
        
        roiAnno = []
        for a in annos:
            if rectIsIn(roiRect, rectClamp(imgRect, a['bbox'])):
                scaleX = self.batchW / roiRect[2]
                scaleY = self.batchH / roiRect[3]
                aClamp = rectClamp(imgRect, a['bbox'])
                aClamp[0] = aClamp[0]-roiRect[0]
                aClamp[1] = aClamp[1]-roiRect[1]
                aClamp = [aClamp[0]*scaleX, aClamp[1]*scaleY, aClamp[2]*scaleX, aClamp[3]*scaleY]
                roiAnno.append(aClamp)
        if len(roiAnno) == 0:
            if len(errorStack) > 10:
                raise Exception(errorStack)
            return self.getFrame(errorStack = errorStack.append(Exception('pivot is disapeaed!', 'roi', roiRect, 'pi', pivotRect, 'img', imgRect)))
        
        #Roi Image
        imgPath = os.path.join(self.imgDir, img['file_name'])
        if not os.path.exists(imgPath):
            print(imgPath)
            raise Exception('file not found')
        imgData = cv2.imread(imgPath)
        imgData = imgData[int(roiRect[1]):int(roiRect[1]+roiRect[3]), int(roiRect[0]):int(roiRect[0]+roiRect[2])]
        imgData = cv2.resize(imgData, dsize=(self.batchW, self.batchH))
        if __name__=='__main__':
            for a in roiAnno:
                x,y,w,h=a
                cv2.rectangle(imgData, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        
        #fill data
        bboxData = np.zeros((5*self.bboxCount, self.outputGridH, self.outputGridW))
        textData = np.zeros((self.textDim*self.bboxCount, self.outputGridH, self.outputGridW))
        styleData = np.zeros((self.styleDim*self.bboxCount, self.outputGridH, self.outputGridW))
        
        latentData = np.random.normal(0, 1, (self.latentDim, self.outputGridH, self.outputGridW))
        
        gridAnno = [[[] for __ in range(self.outputGridH)] for _ in range(self.outputGridW)]
        for a in roiAnno:
            norm = [a[0]/self.batchW, a[1]/self.batchH, a[2]/self.batchW, a[3]/self.batchH]
            norm[0] += norm[2]/2
            norm[1] += norm[3]/2
            x = int(norm[0] / (1 / self.outputGridW))
            y = int(norm[1] / (1 / self.outputGridH))
            if len(gridAnno[x][y])<self.bboxCount:
                gridAnno[x][y].append(norm)
            else:
                pass
        for x in range(self.outputGridW):
            for y in range(self.outputGridH):
                if len(gridAnno[x][y])==0:
                    pass
                else:
                    cellAnnos = gridAnno[x][y]
                    #fill cell
                    textData[:, y, x] = np.random.uniform(-1, 1, (self.textDim*self.bboxCount))
                    styleData[:, y, x] = np.random.uniform(-1, 1, (self.styleDim*self.bboxCount))
                    for i, c in enumerate(cellAnnos):
                        bboxData[i*5+0, y, x] = c[0]
                        bboxData[i*5+1, y, x] = c[1]
                        bboxData[i*5+2, y, x] = c[2]
                        bboxData[i*5+3, y, x] = c[3]
                        bboxData[i*5+4, y, x] = 1
        
        #Expand dim
        frame['bbox']=np.expand_dims(bboxData, axis=0)
        frame['text']=np.expand_dims(textData, axis=0)
        frame['style']=np.expand_dims(styleData, axis=0)
        frame['latent']=np.expand_dims(latentData, axis=0)
        frame['img']=np.expand_dims(imgData, axis=0)
        
        return frame
    
    def batch(self, size):
        """
        - Expected Usage
        batch = data.batch(opt.batch_size)
        batch_image = batch['img']
        batch_bbox = batch['bbox']
        batch_text = batch['text']
        batch_text = batch['style']
        batch_text = batch['latent']
        """
        frames = {'img':[], 'bbox':[], 'text':[], 'style':[], 'latent':[]}
        for _ in range(size):
            frame = self.getFrame()
            frames['img'].append(frame['img'])
            frames['bbox'].append(frame['bbox'])
            frames['text'].append(frame['text'])
            frames['style'].append(frame['style'])
            frames['latent'].append(frame['latent'])
        frames['img'] = np.concatenate(frames['img'], axis=0)
        frames['bbox'] = np.concatenate(frames['bbox'], axis=0)
        frames['text'] = np.concatenate(frames['text'], axis=0)
        frames['style'] = np.concatenate(frames['style'], axis=0)
        frames['latent'] = np.concatenate(frames['latent'], axis=0)
        return frames


# In[10]:


if __name__ =='__main__':
    print("coco text dataset")
    data = coco_text_dataset(ignoreCache=False, batchW=224, batchH=224)
    while True:
        s = time.time()
        batch = data.batch(32)
        elapsed = time.time() - s
        batch_image = batch['img']
        batch_bbox = batch['bbox']
        batch_text = batch['text']
        batch_style = batch['style']
        batch_latent = batch['latent']
        break
#         print(*[batch_bbox[0][i] for i in range(5)], batch_text[0][0], batch_style[0][0], batch_latent[0][0])
#         break
        for i in range(len(batch)):
            img = batch_image[i]
            display.vidshow(img, maxSize=(416,416))
            print(32/elapsed, 's/s', elapsed*1000, 'ms')
            time.sleep(0.03)


# In[11]:


# compile

if __name__ == '__main__':
    if in_notebook():
        get_ipython().system('echo converting...')
        get_ipython().system('ipython nbconvert --to python coco_text_dataset.ipynb')
        get_ipython().system('ls -l')
        get_ipython().system('echo running...')
        get_ipython().system('rem python coco_text_dataset.py')
        get_ipython().system('echo finished.')


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:54:04 2019

@author: Tobias 
"""
import os
import pickle
import numpy as np
import datetime as dt
import pandas as pd
import pandas_datareader as web
import sys
import pdb
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier


class predictor:
    '''
    
    '''
    
    # Class attributes
    
    # Return data
    data        = []
    
    # Publishing dates
    dates       = []
    
    # Publishing dates, 10-K
    dates_k     = []
    
    # Publishing dates, 10-Q
    dates_q     = []
    
    # Company name
    name        = ''
    
    # Date directory
    date_dir    = ''
    
    # Company ticker
    ticker      = ''
    
    # Start date 10-K
    start_k     = []
    
    # Start date 10-Q
    start_q     = []
    
    # Similarity data
    sim_data    = []
    
    # Path for LDA models
    mod_path    = ''
    
    # Years
    years       = []
    
    # Years quarterly
    years_q     = []
    
    # Years annualy 
    years_k     = []
    
    # First valid period with a previous 10-K to use as a benchmark
    first_valid = []
    
    # Number of topics
    num_top     = []
    
    # Data matrix
    X           = []
    
    # Response matrix
    y           = pd.DataFrame()
    
    # Days delayed for returns
    days        = []
    
    # Accuracy 
    accuarcy    = []
    
    # Score
    score       = []
    
    # Return data matrix
    X_ret       = []
    
    def __init__(self,name=None,ticker=None,date_dir=None,f_type=None,
                 algorithm=None, mod_path=None, num_top=None,days=None):
        '''
        Class constructor.
        
        Optional inputs:
            - name      : The EDGAR name of the company. Default is empty.
            - ticker    : The company ticker. Default is empty
            - date_dir  : The directory for the publishing dates. Default is 
                          empty.
            - algorithm : The type of model algorithm. Default is empty. 
                          * Supported algorithms: random forest {'rf'}.
            - f_type    : The filing type. Default is 10-K
            
            - num_top   : The number of topics for the LDA model
            
        Output: 
            - obj: An object of class predictor.
            
        Examples:
            name     = 'BERKSHIRE HATHAWAY INC'
            ticker   = 'BRK-A'
            date_dir = 'C:\\Users\\Tobias\\Dropbox\\Master\\U.S. Data\\Dates Reports U.S'
            mod_path = 'C:\\Users\\Tobias\\Dropbox\\Master\\U.S. Data\\Model U.S'
            num_top  = 40
            days     = 7
        '''
        
        if name:
            self.name     = name
        if f_type:
            self.f_type   = f_type
        if date_dir:
            self.date_dir = date_dir
        if ticker:
            self.ticker   = ticker
        if mod_path:
            self.mod_path = mod_path
        if num_top:
            self.num_top  = num_top
        if days:
            self.days     = days
    
    def get_return_data(self):
        '''
        A method for fetching return data from yahoo
        
        Input:
            - self : An object of class predictor
            
        Output:
            - self : -"-
        '''
        start ='1996-01-01'
        end   ='2019-31-05'
         
        mes=('Checking for existing return data...')
        sys.stdout.write('\r'+mes)  
        
        print('Fetching return data')
        if not self.ticker or not self.name:
            print('Object cannot be empty')
            return
        if self.days: delta = self.days
        else: delta=0
        ticker      = self.ticker
        name        = self.name
        directory_q = self.date_dir + '\\10-Q\\'+ name
        os.chdir(directory_q)
        start  = dt.datetime(1994,1, 1)
        end    = dt.datetime(2019,4,16)
        prices = web.get_data_yahoo(ticker,start=start,end=end)[['Open','Close']]
        with open(os.listdir()[0],'rb') as file:
            dates_q = pickle.load(file)
        directory_k = self.date_dir + '\\10-K\\' + name
        os.chdir(directory_k)
        with open(os.listdir()[0],'rb') as file:
           dates_a = pickle.load(file)
         
        dates = dates_q + dates_a
        publishingdates = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
        stockdates      = [index.strftime('%Y-%m-%d') for index in prices.index]
        stockdates      = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in stockdates]
        
        # Find all indicies where there is a report published
        allDates  = stockdates
        pOp       = prices.Open
        pCl       = prices.Close
        dataOut   = np.zeros(shape=(len(publishingdates),2))
        publishingdates.sort()
        p_delta = [d+dt.timedelta(days=delta) for d in publishingdates]
        l         = 0
        if delta>0:
               for idx, (t_1,t_2) in enumerate(zip(publishingdates,p_delta)):
                    
                    f_1   = [date-t_1 for date in allDates]
                    f_2   = [date-t_2 for date in allDates]
                    arr   = []
                    arr_2 = []
                    for diff_1,diff_2 in zip(f_1,f_2):
                        arr   = np.append(arr,diff_1.days)
                        arr_2 = np.append(arr_2,diff_2.days)
                    zeroInd   = np.where(arr   == 0)[0]
                    zeroInd_2 = np.where(arr_2 == 0)[0]
                    if not zeroInd.size > 0:
                        l = l + 1
                        continue
                    if not zeroInd_2.size > 0:
                        l = l + 1
                        continue
                    ret            = np.log(pCl[zeroInd_2]/pOp[zeroInd][0])
                    dataOut[idx,0] = ret
                    if ret[0]>-1E-8:
                        dataOut[idx,1] = 1
                    else:
                        dataOut[idx,1] = -1   
            
        else:
            for idx, tempDate in enumerate(publishingdates):
                    f   = [date-tempDate for date in allDates]
                    arr = []
                    for diff in f:
                        arr = np.append(arr,diff.days)
                    zeroInd = np.where(arr == 0)[0]
                    if not zeroInd.size > 0:
                        l = l + 1
                        continue
                    ret = np.log(pCl[zeroInd]/pOp[zeroInd])
                    dataOut[idx,0] = ret
                    if ret[0]>-1E-8:
                        dataOut[idx,1] = 1
                    else:
                        dataOut[idx,1] = -1   
            
        years    = [y.split('-')[0] for y in dates]
        years_q  = [y.split('-')[0] for y in dates_q]
        years_k  = [y.split('-')[0] for y in dates_a]
        dates.sort()
        dates_a.sort()
        dates_q.sort()
        years_k.sort()
        years_q.sort()
        years.sort()
        self.data    = dataOut
        self.dates   = dates
        self.dates_k = dates_a
        self.dates_q = dates_q
        self.start_k = dates_a[0]
        self.start_q = dates_q[0]
        self.years   = years
        self.years_q = years_q
        self.years_k = years_k
        resp             = pd.DataFrame(dataOut)
        resp['Datetime'] = publishingdates
        resp   = resp.set_index('Datetime')
        self.y = resp
        return

    def get_similarity(self,numWords=None):
        '''
        '''
        if not numWords:
            numWords = 50   # Set
        numTop   = 40   # Set
        resT     = np.empty((0,numTop))
        res      = np.empty((0,numTop))
        dates    = self.dates
        y_q      = self.years_q
        y_k      = self.years_k
        d_k      = self.dates_k
       
        
        # Identify the first valid year to start estimation
        for t in y_k[1:]:
            num_inds = len([i for i,year in enumerate(y_q) if year in t])
            if num_inds == 3: break
        
        # Identify the last valid year to end estimation
        last_ind = 4*(int(y_k[-2])-int(t)+1)       
        # t now yields the first valid year to start estimation. Find the 
        # corresponding index
        start_ind = y_q.index(t)
        dates     = dates[1+start_ind:start_ind+last_ind+1]
        
        temp = self.y
        temp = temp.iloc[1+start_ind:start_ind+last_ind+1]
        self.y = temp
        
        # Find appropriate starting point for 10-K
        start_y_ind = y_k.index(str(int(t)-1))
        year_dates  = d_k[start_y_ind:]
               
        if numTop:
             mod_path_k = self.mod_path+'\\10-K_'+ str(numTop) +'\\'+self.name
             mod_path_q = self.mod_path+'\\10-Q_'+ str(numTop) +'\\'+self.name
        else:    
            mod_path_k = self.mod_path+'\\10-K_40\\'+self.name
            mod_path_q = self.mod_path+'\\10-K_40\\'+self.name
            
        for idx, year in enumerate(y_k[:-1]):
            os.chdir(mod_path_k)   #self.mod_path+'\\10-K_40\\'+self.name #modelpath_10K
#            ann_mod       = gensim.models.ldamodel.LdaModel.load(ann[0])  
            ann_mod_bench=gensim.models.ldamodel.LdaModel.load(year_dates[idx])  
            d = [d for d in dates if str(int(year)+1) in d]
            for quart in d:      
                if quart in year_dates:
                    os.chdir(mod_path_k)#self.mod_path+'\\10-K_40\\'+self.name
                    lda_model_q = gensim.models.ldamodel.LdaModel.load(quart)  
                else:
                    os.chdir(mod_path_q)#self.mod_path+'\\10-Q_40\\'+self.name
                    lda_model_q = gensim.models.ldamodel.LdaModel.load(quart)  
                    
                mdiff, annotation = lda_model_q.diff(ann_mod_bench, 
                                        distance='jaccard', num_words=numWords)
                resT   = np.empty((0,numTop))
                for ii in range(numTop):
                    g         = mdiff[:,ii].tolist()
                    min_value = min(g)
                    min_index = g.index(min_value)
                    
                    # Assign score
                    resT = np.append(resT, numTop-min_index)
                
                res = np.append(res,resT)
                del resT                   
        length = len(dates)
        resOut = np.reshape(res,(length,numTop))
        X          = pd.DataFrame(resOut)
        self.X     = X
        self.X_ret = np.log(X / X.shift(1))
        return
    
    def get_similarity_ranking(self,numWords=None,fomc_path=None,sim_path=None,reload=None,numTop=None):
        '''
        fomc_path = 'C:\\Users\\Tobias\\Dropbox\\Master\\U.S. Data\\FOMC Monetary Policy Report'
        sim_path  = 'C:\\Users\\Tobias\\Dropbox\\Master\\U.S. Data\\Similarity'
        reload    = 1
        '''
        if not numWords:
            numWords = 50   # Set
        if not numTop:    
            numTop    = 40   # Set
        dates     = self.dates
        y_q       = self.years_q
        d_k       = self.dates_k
        os.chdir(fomc_path)
        with open('fomcdates.pickle', 'rb') as file:
            fomc_dates = pickle.load(file)
        fomc_df   = pd.DataFrame({'date': fomc_dates})
        fomc_df   = fomc_df.set_index('date')
        comp_df   = pd.DataFrame({'date': dates})
        comp_df   = comp_df.set_index('date')
        pairs = []
        for tempDate in fomc_dates:
            t = comp_df.truncate(after=tempDate)
            t = t.tail(1).index.tolist()            
            if t:
                pairs.append([t[0],tempDate])   
                
        score=[]
        date =[]              
        for pair in pairs:
            os.chdir(fomc_path+'\\model_'+str(numTop)) #'\\model'  
            fomc_model = gensim.models.ldamodel.LdaModel.load(pair[1])
            
            if pair[0] in d_k:
                os.chdir(self.mod_path+'\\10-K_' +str(numTop)+'\\'+self.name) #'\\10-K_40\\'
                comp_mod = gensim.models.ldamodel.LdaModel.load(pair[0])
            else: 
                os.chdir(self.mod_path+'\\10-Q_' +str(numTop)+'\\'+self.name) #'\\10-Q_40\\'
                comp_mod = gensim.models.ldamodel.LdaModel.load(pair[0])

            mdiff, annotation = fomc_model.diff(comp_mod, 
                                    distance='jaccard', num_words=numWords)
            
            ###
            mdiff = np.abs(mdiff-1)
            score_t=[]
            for c in range(numTop):
                score_t.append(np.sum(mdiff[:,c]*(1)))
            
            score.append(sum(score_t))
            ###
#            mdiff, annotation = comp_mod.diff(fomc_model,distance='jaccard',
#                                              num_words=numWords)
           # diff = mdiff.reshape((numTop*numTop,1))
           # diff = diff.tolist()
           # diff.sort()
           # ind = np.where(mdiff==diff[0])
           # score.append(np.abs(ind[0][0]-ind[1][0]))
            date.append(pair[1])
        score_out  = pd.DataFrame({'date': date,
                                  self.ticker: score})
        score_out  = score_out.set_index('date')
        self.score = score_out
        if reload:
            os.chdir(sim_path)
            if not 'ranking.pickle' in os.listdir():
                tickers = score_out
                with open('ranking.pickle', 'wb') as f:
                    pickle.dump(tickers, f)
                    return
            with open('ranking.pickle', 'rb') as file:
                tickers = pickle.load(file)
            tickers[self.ticker] = score_out
            with open('ranking.pickle', 'wb') as f:
                pickle.dump(tickers, f)
        return
        
    def predict(self,Rec=None,estimators=None,maxdepth=None,leaf_size=None,randomstate=None,max_features=None,res_path=None,Ret=None):
        '''
        '''
        g
        #estimators   = estimators
        #maxdepth     = 15
        #leaf_size    = 1
        #randomstate  = None
        #max_features = 'auto'
        np.random.seed(1337)
        
        
        if Ret:
            if Rec:
                result = []
                X      = self.X_ret.iloc[1:,:]
                y      = self.y.iloc[1:,1]
                n      = len(X)
                y_p    = []
                for recursion in reversed(range(1,Rec+1)):
                    X_train = X.iloc[:-1-recursion]
                    y_train = y.iloc[:-1-recursion]
                    X_test  = X.iloc[[n-recursion ]]
                    y_test  = y.iloc[[n-recursion]]
                    clf=RandomForestClassifier(n_estimators     = estimators,
                                               max_depth        = maxdepth,
                                               min_samples_leaf = leaf_size,
                                               random_state     = randomstate,
                                               max_features     = max_features)
                    
                    clf.fit(X_train,y_train)        
                    y_pred = clf.predict(X_test)
                    result.append( metrics.accuracy_score(y_test, y_pred))
                    y_p.append(y_pred[0])
                y_pred = y_p
                y_ret  = self.y.iloc[:,0]
                y_ret  = y_ret[-Rec:]
                self.accuracy = np.sum(result)/(Rec) 
                print('Accuracy: ',np.sum(result)/(Rec))    
                
                
            else:
                X      = self.X_ret.iloc[1:,:]
                X = X.abs() ##############################
                y      = self.y.iloc[1:,1]
                X_train, X_test = train_test_split(X,shuffle=False,test_size=0.06)       
                y_train, y_test = train_test_split(y,shuffle=False,test_size=0.06)  
                clf=RandomForestClassifier(n_estimators     = estimators,
                                           max_depth        = maxdepth,
                                           min_samples_leaf = leaf_size,
                                           random_state     = randomstate,
                                           max_features     = max_features)  
                                         
                clf.fit(X_train,y_train)        
                y_pred=clf.predict(X_test)
                self.accuracy = metrics.accuracy_score(y_test, y_pred)                
                _ , y_ret = train_test_split(self.y.iloc[1:,0],shuffle=False,
                                                                test_size=0.06)
                if y_test[0]*y_pred[0]<0:
                    print('Returning...')
                    #print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
                    ret = np.sum(np.multiply(y_ret[1:],-1*y_pred[1:])+y_ret[0]*y_pred[0])/len(y_ret)
                    r   = pd.DataFrame({self.ticker: [ret,self.accuracy]})
                else:
                    print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
                
                    _ , y_ret = train_test_split(self.y.iloc[1:,0],shuffle=False,
                                                                  test_size=0.06)
                    ret = np.sum(np.multiply(y_ret,y_pred))/len(y_ret)
                    r   = pd.DataFrame({self.ticker: [ret,self.accuracy]})
                   

                
        else:
            if Rec:
                result = []
                X      = self.X
                y      = self.y.iloc[:,1]
                n      = len(X)
                y_p    = []
                for recursion in reversed(range(1,Rec+1)):
                    X_train = X.iloc[:-1-recursion]
                    y_train = y.iloc[:-1-recursion]
                    X_test  = X.iloc[[n-recursion ]]
                    y_test  = y.iloc[[n-recursion]]
                    clf=RandomForestClassifier(n_estimators     = estimators,
                                               max_depth        = maxdepth,
                                               min_samples_leaf = leaf_size,
                                               random_state     = randomstate,
                                               max_features     = max_features)
                    
                    clf.fit(X_train,y_train)        
                    y_pred = clf.predict(X_test)
                    result.append( metrics.accuracy_score(y_test, y_pred))
                    y_p.append(y_pred[0])
                y_pred = y_p
                y_ret  = self.y.iloc[:,0]
                y_ret  = y_ret[-Rec:]
                self.accuracy = np.sum(result)/(Rec) 
                print('Accuracy: ',np.sum(result)/(Rec))    
                
                
            else:
                X               = self.X
                y               = self.y.iloc[:,1]
                X_train, X_test = train_test_split(X,shuffle=False,test_size=0.06)       
                y_train, y_test = train_test_split(y,shuffle=False,test_size=0.06)  
                clf=RandomForestClassifier(n_estimators     = estimators,
                                           max_depth        = maxdepth,
                                           min_samples_leaf = leaf_size,
                                           random_state     = randomstate,
                                           max_features     = max_features)  
                                         
                clf.fit(X_train,y_train)        
                y_pred=clf.predict(X_test)
                self.accuracy = metrics.accuracy_score(y_test, y_pred)
                
                # TEST
                _ , y_ret = train_test_split(self.y.iloc[:,0],shuffle=False,
                                                                  test_size=0.06)
               # if y_test[0]*y_pred[0]<0:
              #     print('Returning...')
              #     #print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
             #      ret = np.sum(np.multiply(y_ret[1:],-1*y_pred[1:])+y_ret[0]*y_pred[0])/len(y_ret)
             #      r   = pd.DataFrame({self.ticker: [ret,self.accuracy]})
             #   else:
                print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
                _ , y_ret = train_test_split(self.y.iloc[:,0],shuffle=False,
                                                              test_size=0.06)
                ret = np.sum(np.multiply(y_ret,y_pred))/len(y_ret)
                r   = pd.DataFrame({self.ticker: [ret,self.accuracy]})
       
        
        os.chdir(res_path)
        if Rec:
            name = 'results.'+str(self.num_top)+ '.' +str(Rec)+ '.pickle'
            if not name in os.listdir():
                with open(name, 'wb') as f:
                    pickle.dump(r, f)
                    return                
                
            with open(name,'rb') as file:
                results = pickle.load(file)
            if results.empty:
                results = r
            else:    
                results[self.ticker] =  r
            with open(name, 'wb') as f:
                pickle.dump(results, f)
            return
        else:
            name = 'results.'+str(self.num_top)+ '.pickle'
            if not name in os.listdir():
                with open(name, 'wb') as f:
                    pickle.dump(r, f)
                    return
                
            with open(name,'rb') as file:
                results = pickle.load(file)
            if results.empty:
                results = r
            else:     
                results[self.ticker] =  r
            with open(name, 'wb') as f:
                pickle.dump(results, f)
                
            return
           

    
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

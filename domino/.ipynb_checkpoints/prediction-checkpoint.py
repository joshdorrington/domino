import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def CV_None(predictors,target,**cv_kwargs):
    return [[predictors,target,predictors,target]]

def CV_drop1year(predictors,target,**cv_kwargs):
    
    X,y=xr.align(predictors,target)
    years=np.unique(X['time.year'])
    iters=[]
    for t in years:
        it=[]
        it.append(X.where(X['time.year']!=t).dropna('time'))
        it.append(y.where(y['time.year']!=t).dropna('time'))
        it.append(X.where(X['time.year']==t).dropna('time'))
        it.append(y.where(y['time.year']==t).dropna('time'))
        iters.append(it)
    return iters

def CV_n_chunks(predictors,target,N=2):
    
    X,y=xr.align(predictors,target)
    L=len(y)
    l=L//N
    iters=[]
    for n in range(N):
        
        test_times=np.arange(n*l,(n+1)*l)
        train_times=np.arange(L)
        train_times=np.delete(train_times,test_times)
        
        X_train=X.isel(time=train_times)
        y_train=y.isel(time=train_times)                             
        X_test=X.isel(time=test_times)
        y_test=y.isel(time=test_times)
        
        iters.append([X_train,y_train,X_test,y_test])
    return iters

def log_regression_model(X_train,y_train,X_test,**model_kwargs):
    
    LR=LogisticRegression(**model_kwargs)
    LR=LR.fit(X_train,y_train)
    
    det_pred=LR.predict(X_test)
    prob_pred=LR.predict_proba(X_test)
    return LR,det_pred,prob_pred

def blank_score(det_pred,prob_pred,target):
    
    if det_pred is not None:
        assert det_pred.shape[0]==target.shape[0]
    if prob_pred is not None:
        assert prob_pred.shape[0]==target.shape[0]
    return len(target)

def ROC_AUC(det_pred,prob_pred,target,**score_kwargs):
    if len(np.unique(target))==2:
        prob_pred=prob_pred[:,1]
    return roc_auc_score(target,prob_pred,**score_kwargs)

class PredictionTest(object):
    
    def __init__(self,predictors,predictand):
        
        self.predictand=self._predictand_to_dataarray(predictand)
        self.predictors=self._predictors_to_dataset(predictors)
        
        
        self.predefined_models=dict(
            logregression=log_regression_model
        )
        
        self.predefined_cv_methods=dict(
            nchunks=CV_n_chunks,
            drop_year=CV_drop1year
        )
        
        self.predefined_scores=dict(
            sklearn_roc_auc=ROC_AUC,
            test=blank_score
        )
        return
    
    def __repr__(self):
        return 'A PredictionTest object'
    def __str__(self):
        return self.__repr__
    
    def _predictand_to_dataarray(self,predictand):
        return predictand
    
    def _predictors_to_dataset(self,predictors):
        return predictors
    
    def categorical_prediction(self,model,score='sklearn_roc_auc',cv_method=None,predictor_variables='univariate',keep_models=False,model_kwargs={},cv_kwargs={},score_kwargs={}):
        
        if np.ndim(predictor_variables)==1:
            score_labels=predictor_variables
            
        elif np.ndim(predictor_variables)==2:
            score_labels=[f'predictor_set_{i+1}' for i in range(len(predictor_variables))]
            
        elif predictor_variables =='univariate':
            predictor_variables = list(self.predictors.data_vars)
            score_labels=list(self.predictors.data_vars)
        elif predictor_variables=='all':
            predictor_variables=[list(self.predictors.data_vars)]
            score_labels=['all']
        else:
            raise(ValueError('predictor variables must be one of ["univariate","all",1-D array, 2-D array]'))
        
        if type(score)==str:
            score=self.predefined_scores[score]
        
        predictors=self.predictors
        target=self.predictand
        predictors,target=xr.align(predictors.dropna('time'),target.dropna('time'))
        
        #train_pred,train_target,test_pred,test_target
        cv_iterator=self._split_data(predictors,target,cv_method,**cv_kwargs)
        
        score_da=[]
        model_da=[]
        for split_data in cv_iterator:
            scores={}
            models={}
            train_preds,train_target,test_preds,test_target=split_data
            
            for label,variable in zip(score_labels,predictor_variables):
                
                train_pred=train_preds[variable]
                test_pred=test_preds[variable]
                
                #univariate
                if type(variable)==str:
                    
                    train_pred=np.array(train_pred).reshape([-1,1])
                    test_pred=np.array(test_pred).reshape([-1,1])

                #multivariate
                else:
                    train_pred=np.array(train_pred.to_array('var').transpose(*train_pred.dims,'var'))
                    test_pred=np.array(test_pred.to_array('var').transpose(*test_pred.dims,'var'))

                fitted_model,det_pred,prob_pred=self._fit_model_and_predict(model,train_pred,train_target,test_pred,**model_kwargs)
                test_score=score(det_pred,prob_pred,test_target,**score_kwargs)
                scores[label]=test_score
                if keep_models:
                    models[label]=fitted_model
            score_da.append(xr.Dataset(scores))
            model_da.append(models)
        score_da=xr.concat(score_da,'cv')
        if keep_models:
            return score_da,model_da
        else:
            return score_da
    
    def _fit_model_and_predict(self,model,train_pred,train_target,test_pred,**kwargs):
        
        if type(model)==str:
            model=self.predefined_models[model]
        
        train_pred=np.array(train_pred)
        train_target=np.array(train_target)
        test_pred=np.array(test_pred)
        self._verify_model_inputs(train_pred,train_target,test_pred)
        output=model(train_pred,train_target,test_pred,**kwargs)
        self._verify_model_output(output,test_pred)
        return output
    
    def _verify_model_inputs(self, train_pred,train_target,test_pred,**model_kwargs):
    
        assert train_pred.ndim==2
        T1,N=train_pred.shape

        assert test_pred.ndim==2
        T2,M=test_pred.shape
        assert N==M

        assert train_target.ndim==1
        T=len(train_target)
        assert T==T1
        return 

    def _verify_model_output(self,output,test_pred):
        
        T,N=test_pred.shape
        assert len(output)==3
        model,det_pred,prob_pred=output

        assert (det_pred is not None)|(prob_pred is not None)

        if det_pred is not None:
            assert det_pred.shape[0]==T
            assert np.ndim(det_pred)==1
        if prob_pred is not None:
            assert prob_pred.shape[0]==T
            assert np.ndim(prob_pred)==2
        return
    
    def _split_data(self,predictors,target,cv_method,**cv_kwargs):
        if cv_method is None:
            cv_method=CV_None
        elif type(cv_method)==str:
            cv_method=self.predefined_cv_methods[cv_method]
        
        cv_iterator=cv_method(predictors,target,**cv_kwargs)
        return cv_iterator
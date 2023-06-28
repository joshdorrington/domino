import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def CV_None(predictors,target,**cv_kwargs):
    """No cross validation: train and test data are the same."""
    return [[predictors,target,predictors,target]]

def CV_drop1year(predictors,target,**cv_kwargs):
    """Cross validation is performed by dropping out single years of data from the training set to use as test data. This is repeated for all years with non-nan data."""

    X,y=xr.align(predictors.dropna('time',how='all'),target.dropna('time',how='all'))
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
    """The dataset is split into *N* chunks and each is successively held out as test data."""
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

def CV_manual(train_pred,train_targ,test_pred=None,test_targ=None):
    """Separate training and test datasets are passed manually."""
    if test_pred is None or test_targ is None:
        raise(ValueError('Both test_pred and test_targ must be specified when using CV_manual'))
    return [[train_pred,train_targ,test_pred,test_targ]]

def log_regression_model(X_train,y_train,X_test,**model_kwargs):
    """A wrapper around sklearn's *LogisticRegression* class, which accepts the same *model_kwargs*"""
    LR=LogisticRegression(**model_kwargs)
    LR=LR.fit(X_train,y_train)
    
    det_pred=LR.predict(X_test)
    prob_pred=LR.predict_proba(X_test)
    return LR,det_pred,prob_pred

def blank_score(det_pred,prob_pred,target,**score_kwargs):
    """@private"""
    if det_pred is not None:
        assert det_pred.shape[0]==target.shape[0]
    if prob_pred is not None:
        assert prob_pred.shape[0]==target.shape[0]
    return len(target)

def ROC_AUC(det_pred,prob_pred,target,**score_kwargs):
    """Computes the ROC AUC score, as implemented in sklearn, and accepting the same *score_kwargs*"""
    #No ROC_AUC for a variable with only one value
    if len(np.unique(target))==1:
        return np.nan
    if len(np.unique(target))==2:
        prob_pred=prob_pred[:,1]
        
    
    return roc_auc_score(target,prob_pred,**score_kwargs)

class PredictionTest(object):
    """
    Supports quick and simple testing of the predictive power of predictor variables stored in an xarray.Dataset applied to a target variable stored in an xarray.DataArray.
    Supports different predictive models, cross validation approaches and skill scores within a straightforward API.
    
    **Arguments**
    
    *predictors*
    
    An xarray Dataset of scalar predictors
    
    *predictand*
    
    An xarray DataArray with a shared coord to *predictors*.
    """
    def __init__(self,predictors,predictand,tolerate_empty_variables=False):
        """Initialise a PredictionTest object"""
        
        self.predictand=self._predictand_to_dataarray(predictand)
        """@private"""
        
        self.predictors=self._predictors_to_dataset(predictors)
        """@private"""

        
        self.predefined_models=dict(
            logregression=log_regression_model
        )
        """@private"""
        
        self.predefined_cv_methods=dict(
            nchunks=CV_n_chunks,
            drop_year=CV_drop1year,
            manual=CV_manual
        )
        """@private"""
        
        self.predefined_scores=dict(
            sklearn_roc_auc=ROC_AUC,
            test=blank_score
        )
        """@private"""
        
        self.computed_scores=None
        """@private"""
        
        self.tol_empty=tolerate_empty_variables
        """@private"""
        return
    
    def __repr__(self):
        return 'A PredictionTest object'
    def __str__(self):
        return self.__repr__
    
    def _predictand_to_dataarray(self,predictand):
        return predictand
    
    def _predictors_to_dataset(self,predictors):
        return predictors
    
    def _handle_potentially_empty_variable_list(self,vlist):
        if len(vlist)>0:
            return 0
        else:
            if self.tol_empty:
                return 1
            raise(ValueError('Empty variable list passed, but tolerate_empty_variables=False was set in PredictionTest.init'))

    def categorical_prediction(self,model,score='sklearn_roc_auc',cv_method=None,predictor_variables='univariate',keep_models=False,model_kwargs={},cv_kwargs={},score_kwargs={}):
        
        """
        
        **Arguments**
        
        *model*
        
        A function with the following signature:
        
        Input: Three numpy arrays of shape [T1,N], [T1], [T2,N] (train_predictors, train_target, and test_predictors respectively), and an arbitrary number of keyword arguments.
        
        Output: *model* (A scikit-learn-like fitted model object),*det_pred* (a numpy array of shape [T2] with deterministic predictions) and *prob_pred* (a numpy array of shape [T2,M] with probabilistic predictions, where M is the number of unique values in *train_target* and each element predicts the probability of a given class). Any output can instead be replaced with None, but at least one of *det_pred* and *prob_pred* must not be None.
            
        **Optional arguments**
        
        *score*
        
        A string specifying a predefined skill score (currently only 'sklearn_roc_auc') or a function with the signature:
        
        Input: det_pred (a numpy array of shape [T2]), prob_pred (a numpy array of shape [T2,M], a truth series (a numpy array of shape [T2]), and an arbitrary number of keyword arguments.
        Output: a scalar skill score.
            
        *cv_method*
        A string specifying a predefined cross-validation method, a custom cross-validation function with corresponding signature, or None, in which case no cross-validation is used (the training and test datasets are the same). Predefined cross-validation methods are:
        
        *nchunks* - Divide the dataset into *n* chunks, using each as the test dataset predicted by the other *n*-1 chunks, to produce *n* total skill estimates. *n* defaults to 2, and is specified in *cv_kwargs*
        
        *drop_year* - Split the dataset by calendar year, using each year as the test dataset predicted by the remaining years.
        *manual* - Treat *predictors* and *predictand* as training data. Test data must be passed explicitly via *cv_kwargs* as *test_pred* and *test_targ*.
        
        If a custom function is passed it must have the following signature:
        Input: predictors (a Dataset), target (a DataArray), and an arbitrary number of keyword arguments.
        
        Output: A train predictor Dataset, a train target DatArray, a test predictor Dataset, and a test target DataArray.

        *predictor_variables*
        
        If 'univariate' all variables in *PredictionTest.predictors* are used individually to predict *PredictionTest.predictand*.
        
        If 'all' all variables in *PredictionTest.predictors* are used collectively to predict *PredictionTest.predictand*.
        
        If a 1D array of variable names in *PredictionTest.predictors*, each specified variable is used individually to predict *PredictionTest.predictand*.
        
        If a 2D array of iterables over variable names in *PredictionTest.predictors*, each specified combination of variables is used to predict *PredictionTest.predictand*.
            
        *keep_models*
        If True, a dictionary of arrays of fitted models is returned, corresponding to each variable combination and cross validated model that was computed.
        
        *model_kwargs*
        
        A dictionary of keyword arguments to pass to *model*
        *cv_kwargs*
        
        A dictionary of keyword arguments to pass to *cv_method*

        *score_kwargs*
        
        A dictionary of keyword arguments to pass to *score*
        
        **Returns**
        
        If keep_models is False:
        
        returns *score_da*, a Dataset of skill scores for each prediction experiment, with a cross_validation coordinate.
        
        if keep_models is True:
        
        return (*score_da*,*model_da*)
        """
        if np.ndim(predictor_variables)==1:
            el_type=[np.ndim(element) for element in predictor_variables]
            
            if len(np.unique(el_type))==1 and np.unique(el_type)==1:#a list of ragged lists
                score_labels=[f'predictor_set_{i+1}' for i in range(len(predictor_variables))]
                
            else:#just a list
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
        predictors,target=xr.align(predictors.dropna('time',how='all'),target.dropna('time',how='all'))
        
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
        self.computed_scores=score_da

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
    
    def add_score_to_index_metadata(self,indices,label='score',raise_on_missing_var=False,reduce_func=np.nanmean):
        
        """ Annotate a Dataset of indices with computed skill scores by adding them to the attributes of each DataArray in the Dataset.
        
        **Arguments**
        
        *indices*
        
        *An xarray.Dataset of indices*.
        
        **Optional arguments**
        
        *label*
        
        A string determining the name of the added attribute key.
        
        *raise_on_missing_var*
        
        A boolean, determining if an error is raised if not all variables present in the computed skill scores are present in the indices.
        
        *reduce_func*
        
        The function used to reduce the 'cv' vector of skill scores to a single value. Default is the mean average, ignoring nans. To add the entire vector of scores for different cross validations, pass *reduce_func*=lambda x: x
        
        """
        s=self.computed_scores
        if s is None:
            raise(ValueError('No scores computed.'))
        for var in s.data_vars:
            
            score=s[var].values
            try:
                indices[var].attrs[label]=reduce_func(score)
            except:
                if raise_on_missing_var:
                    raise(ValueError(f'Key {var} present in self.computed_scores but not in indices.'))
                pass
        return
        
        
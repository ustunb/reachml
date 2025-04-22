from abc import ABC, abstractmethod

import shap
import lime
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.ext.data import BinaryClassificationDataset


class ModelExplainer(ABC):
    """
    wrapper class used to generate explanations
    """
    EPSILON = 0.1

    def __init__(self, model, data, **kwargs):
        """
        """
        assert isinstance(data, BinaryClassificationDataset)

        self.model = model
        self.data = data
        self.feat_names = data._names.X
        self.exp = None

    @abstractmethod
    def _generate_exp(self, **kwargs):
        pass

    def _gen_exp_helper(self, rows):
        if isinstance(rows, np.ndarray) and len(rows.shape) > 2:
            # random forests have probability output for shap
            # shap output will be (n_samples, n_features, 2)
            # take probability of prediction -1 (adverse outcome)
            # may need to fix if api is used outside of recourse context
            rows = rows[:, :, 0] 
        exp_df = pd.DataFrame(rows).fillna(0).sort_index(axis=1)
        exp_df = exp_df.rename(
            {idx: feat for idx, feat in enumerate(self.feat_names)},
            axis=1
        )

        self.exp = {
            'values': exp_df,
            'probas': self.model.predict_proba(self.data.U),
        }

    def get_explanations(self, overwrite=False):
        """
        """
        if self.exp is None or overwrite:
            self._generate_exp()

        return self.exp


class SHAP_Explainer(ModelExplainer):
    """
    wrapper class used to generate explanations
    """

    def __init__(self, model, data, **kwargs):
        """
        """
        super().__init__(model, data, **kwargs)
        self.scaler = kwargs.get('scaler', None)
        
        masker = self.data.X if self.scaler is None \
            else self.scaler.transform(self.data.X)

        # f = lambda x: self.model.predict_proba(x)[:, 0]
        self.explainer = shap.Explainer(
            model=model, 
            masker=masker
        )  
        self.explainer_name = 'SHAP'

    def _generate_exp(self, **kwargs):
        """
        """ 
        exp_input = self.data.U if self.scaler is None \
            else self.scaler.transform(self.data.U)
        self._gen_exp_helper(self.explainer(exp_input).values)


class LIME_Explainer(ModelExplainer):
    """
    wrapper class used to generate explanations
    """

    def __init__(self, model, data, **kwargs):
        """
        """
        super().__init__(model, data, **kwargs)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.data.U, 
                mode='classification', 
                feature_names=self.data._names.X,
                discretize_continuous=False,
            )
        self.explainer_name = 'LIME'

    def _generate_exp(self, **kwargs):
        """
        """
        # lime doesn't support batch processing, so we need to loop through the data
        # no regularization with num_features = len(self.feat_names)
        wt_series_lst = []  # list of pandas series of weight vectors indexed by feature name
        intercept_lst = []
        for x in tqdm(self.data.U):
            x_exp = self.explainer.explain_instance(
                        data_row=x, 
                        predict_fn=self.model.predict_proba, 
                        num_features=len(self.feat_names),
                        num_samples=1000
                    )
            weights = np.array(x_exp.as_map()[1])
            intercept = x_exp.intercept[1]
            
            # convert to pandas series (index: feature index)
            x_exp_series = pd.Series(weights[:,1], index=weights[:,0])
            
            wt_series_lst.append(x_exp_series)
            intercept_lst.append(intercept)

        self._gen_exp_helper(wt_series_lst)

        # unique to LIME_Explainer (since LIME returns a linear model)
        self.exp['intercept'] = intercept_lst

        
class actionAwareExplainer(ModelExplainer):
    """
    """

    def __init__(self, og_explainer, action_set, **kwargs):
        """
        """
        # assert isinstance(og_explainer, ModelExplainer)
        self.og_explainer = og_explainer

        super().__init__(og_explainer.model, og_explainer.data, **kwargs)
        self.action_set = action_set
        self.explainer_name = og_explainer.explainer_name + '_actionAware'

    def _generate_exp(self, **kwargs):
        """
        """
        og_exp = self.og_explainer.get_explanations()['values'].copy()
        non_act_feats = ~pd.Series(self.action_set.actionable).to_numpy()

        # non actionable features are set to 0
        og_exp.loc[:, non_act_feats] = 0
        
        self.exp = {
            'values': og_exp,
            'probas': self.og_explainer.get_explanations()['probas']
        }
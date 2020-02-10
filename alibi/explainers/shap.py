import copy
import logging
import scipy.sparse
import shap

import numpy as np
import pandas as pd

from alibi.explainers.base import Explanation, Explainer
from shap.common import DenseData, DenseDataWithIndex
from typing import Callable, Union, List, Sequence, Optional
from alibi.utils.wrappers import methdispatch

logger = logging.getLogger(__name__)


SHAP_PARAMS = ['link', 'group_names', 'groups', 'weights', 'kwargs']

DEFAULT_META_SHAP = {
    "name": None,
    "type": "blackbox,",
    "algorithm": "kernel",
    "explanations": ["global"],
    "params": {
        "summarised": False,
        "grouped": False,
        "transposed": False,
    }
}  # type: dict

DEFAULT_DATA_SHAP = {
    "shap_values": [],
    "expected_values": [],
}  # type: dict


BACKGROUND_WARNING_THRESHOLD = 300


class KernelExplainer(Explainer):

    def __int__(self,
                predictor: Callable,
                background_data: Union[shap.common.Data, pd.DataFrame, pd.Series, np.ndarray, scipy.sparse.spmatrix],
                summarise_background: bool = False,
                n_background_samples: int = 300,
                link: str = 'identity',
                group_names: Sequence = None,
                groups: List[Sequence[int]] = None,
                weights:  Sequence[Union[float, int]] = None,
                **kwargs,
                ):
        """
        A wrapper around the shap.KernelExplainer class. This extends the current shap library functionality
        by allowing the user to specify variable groups in order to deal with one-hot encoded categorical
        variables. The user can also specify whether to aggregate the shap values estimate for the encoded levels
        of categorical variables during the explain call.

        Parameters
        ----------
        predictor
            A callable that takes as an input a samples x features array and outputs a samples x n_outputs
            outputs. The n_outputs should represent model output in margin space. If the model outputs
            probabilities, then the link should be set to 'logit' to ensure correct force plots.
        background_data
            Data used to estimate feature contributions and baseline values for force plots.
        summarise_background
            A large background dataset impacts the runtime and memory footprint of the algorithm. By setting
            this argument to True, only n_background_samples from the provided data are selected. If group_names or
            groups arguments are specified, the algorithm assumes that the data contains categorical variables so
            the records are selected uniformly at random. Otherwise, shap.kmeans (a wrapper around sklearn kmeans
            implementation) is used for selection.
        link
            Valid values are 'identity' or 'logit'. A generalized linear model link to connect the feature
            importance values to the model output. Since the feature importance values, phi, sum up to the
            model output, it often makes sense to connect them to the ouput with a link function where
            link(outout) = sum(phi). If the model output is a probability then the LogitLink link function
            makes the feature importance values have log-odds units. Note that shap operates in margin space
            so in order to obtain correct force plot visualisations the link='logit' must be specified if
            the model outputs probabilities.
        group_names:
            If specified, this array is used to treat groups of features as one during feature perturbation.
            This feature can be useful, for example, to treat encoded categorical variables as one and can
            result in computational savings (this may require adjusting the nsamples parameter).
        groups:
            A list containing sublists specifying the indices of features belonging to the same group.
        weights:
            A sequence or array of weights. This is used only if grouping is specified and assigns a weight
            to each point in the dataset.
        kwargs:
            Expected keyword arguments include "keep_index" and "keep_index_ordered" should be used if a
            data frame containing an index column is passed to the algorithm.
        """

        super(KernelExplainer, self).__int__()

        # user has specified variable groups
        use_groups = groups is not None or group_names is not None
        self.use_groups = use_groups
        # if the user specifies groups but no names, the groups are automatically named
        self.create_group_names = False
        # background_data gets transposed if sum of groups entries matches first dimension
        self.transposed = False
        # sums up shap values for each level of categorical var
        self.summarise_result = False
        # selects a subset of the background data to avoid excessively slow runtimes
        self.summarise_background = summarise_background
        if self.summarise_background:
            background_data = self._summarise_background(background_data, n_background_samples)

        # check user inputs to provide helpful errors and warnings
        self._check_inputs(background_data, group_names, groups, weights)
        if self.create_group_names:
            group_names = ['group_i'.format(i) for i in range(len(groups))]
        if weights is None:
            weights = []

        # perform grouping if requested by the user
        self.background_data = self._get_data(background_data, group_names, groups, weights, **kwargs)
        self.explainer = shap.KernelExplainer(predictor, self.background_data, link=link)  # type: shap.KernelExplainer
        if not self.explainer.vector_out:
            logging.warning(
                "Predictor returned a scalar value. Ensure the output represents a probability or decision score "
                "as opposed to a classification label!"
            )

        # update metadata
        self.meta.update(DEFAULT_META_SHAP)
        self.meta['name'] = self.explainer.__class__.__name__
        called_params = locals()
        for param in called_params.keys():
            if param in SHAP_PARAMS:
                self.meta.update([(param, called_params[param])])
        self.meta['params'].update([("grouped", self.use_groups), ("transposed", self.transposed)])

    def _check_inputs(self,
                      background_data: Union[shap.common.Data, pd.DataFrame, np.ndarray, scipy.sparse.spmatrix],
                      groups: Optional[List[Sequence[int]]],
                      group_names: Optional[Sequence],
                      weights: Optional[Sequence],
                      ) -> None:
        """
        If user specifies parameter grouping, then we check input is correct or inform
        them if the settings they put might not behave as expected.

        Parameters
        ---------
            See constructor.
        """

        # if the user specified this object, checks are redundant
        if isinstance(background_data, shap.common.Data):
            return

        if self.background_data.shape[0] > BACKGROUND_WARNING_THRESHOLD:
            if not self.summarise_background:
                logging.warning(
                    "Large datasets can cause slow runtimes for shap. The background dataset "
                    "provided has {} records. Consider passing a subset or allowing the algorithm "
                    "to automatically summarize the data by setting the summarise_background=True."
                )

        if group_names and not groups:
            logging.info(
                "Specified group_names but no corresponding sequence 'groups' with indices "
                "for each group was specified. All groups will have len=1."
            )

        if groups and not group_names:
            logging.warning("No group names specified but groups specified! Automatically"
                            "assigning 'group_' name for every feature!")
            self.create_group_names = True

        if groups:
            if not isinstance(groups[0], Sequence):
                raise TypeError(
                    "groups should be specified as List[Sequence[int]] where each "
                    "sublist represents a group and int represent group instance. "
                    "Specified group elements have type {}".format(type(groups[0]))
                )

            expected_dim = sum(len(g) for g in groups)
            actual_dim = background_data.shape[1]
            if expected_dim != actual_dim:
                if background_data.shape[0] == expected_dim:
                    logging.warning("The sum of the group indices list did not match the "
                                    "data dimension along axis=1 but matched dimension "
                                    "along axis=0. The data will be transposed!")
                    self.transposed = True
                else:
                    msg = "The sum of the group sizes specified did not match the number of features. " \
                          "Sum of group sizes: {}. Number of features: {}."
                    raise ValueError(msg.format(expected_dim, actual_dim))

        if weights:
            data_dim = background_data.shape[0]
            feat_dim = background_data.shape[1]
            weights_dim = len(weights)
            if data_dim != weights_dim:
                if not (feat_dim == weights_dim and self.transposed):
                    msg = "The number of weights specified did not match data dimension. " \
                          "Number of weights: {}. Number of datapoints: {}"
                    raise ValueError(msg.format(weights_dim, data_dim))

    def _summarise_background(self,
                              background_data: Union[shap.common.Data, pd.DataFrame, np.ndarray, scipy.sparse.spmatrix],
                              n_background_samples: int,
                              ) -> Union[shap.common.Data, pd.DataFrame, np.ndarray, scipy.sparse.spmatrix]:
        """
        Summarises the background data to n_background_samples in order to reduce the computational cost. If the
        background data is a shap.common.Data object, no summarisation is performed.

        Parameters
        ----------
            See constructor.

        Returns
        -------
            If the user has specified grouping, then the input object is subsampled and and object of the same
            type is returned. Otherwise, a shap.common.Data object containing the result of a kmeans algorithm
            is wrapped in a shap.common.DenseData object and returned. The samples are weighted according to the
            frequency of the occurence of the clusters in the original data.
        """

        if isinstance(background_data, shap.common.Data):
            logging.warning(
                "Received option to summarise the data but the background_data object was an "
                "instance of shap.common.Data. No summarisation will take place!")
            return background_data
        elif self.use_groups:
            return shap.sample(background_data, nsamples=n_background_samples)
        else:
            logging.info(
                "When summarising with kmeans, the samples are weighted in proportion to their "
                "cluster occurrence frequency. Please specify a different weighting of the samples "
                "through the by passing a weights of len=n_background_samples to the constructor!"
            )
            return shap.kmeans(background_data, n_background_samples)

    @methdispatch
    def _get_data(self,
                  background_data: Union[shap.common.Data, pd.DataFrame, pd.Series, np.ndarray, scipy.sparse.spmatrix],
                  group_names: Sequence,
                  groups: List[Sequence[int]],
                  weights: Sequence[Union[float, int]],
                  **kwargs,
                  ):
        """
        Parameters
        ----------
            See constructor.
        """
        pass

    @_get_data.register(shap.common.Data)
    def _(self, background_data, *args) -> shap.common.Data:
        """
        Initialises background data if user passes a shap.common.Data object as input.
        Input  is returned as this is a native object to shap.
        """

        _, _, weights = args
        if weights and self.summarise_background:
            cluster_weights = background_data.weights
            n_cluster_weights = len(cluster_weights)
            n_weights = len(weights)
            if n_cluster_weights != n_weights:
                msg = "The number of weights vector provided ({}) did not match the number of " \
                      "summary data points ({}). The weights provided will be ignored!"
                logging.warning(msg.format(n_weights, n_cluster_weights))
            else:
                background_data.weights = weights

        return background_data

    @_get_data.register(pd.core.frame.Series)
    def _(self, background_data, *args) -> Union[shap.common.Data, pd.core.frame.Series]:
        """
        Initialises background data if user passes a pandas Series object as input.
        Original object is returned as this is initialised internally by shap is there
        is no group structure specified. Otherwise the a shap.common.DenseData object
        is initialised.
        """

        _, groups, _ = args
        if self.use_groups:
            return DenseData(
                background_data.values.reshape(1, len(background_data)),
                list(background_data.index),
                groups,
            )

        return background_data

    @_get_data.register(scipy.sparse.spmatrix)
    def _(self, background_data, *args) -> Union[shap.common.Data, scipy.sparse.spmatrix]:
        """
        Initialises background data if user passes a sparse matrix as input. If the
        user specifies feature grouping, then the sparse array is converted to a dense
        array. Otherwise, the original array is returned and handled internally by shap
        library.
        """

        if self.use_groups:
            logging.warning(
                "Grouping is not currently compatible with sparse matrix inputs. "
                "Converting background data sparse array to dense matrix."
            )
            background_data = background_data.toarray()
            return DenseData(
                background_data,
                *args,
            )

        return background_data

    @_get_data.register(pd.core.frame.DataFrame)
    def _(self, background_data, *args, **kwargs) -> Union[shap.common.Data, pd.core.frame.DataFrame]:
        """
        Initialises background data if user passes a pandas.core.frame.DataFrames as input.
        If the user has specified groups and given a data frame, initialise shap.common.DenseData
        explicitly as this is not handled by shap library internally. Otherwise, data initialisation,
        is left to the shap library.
        """

        _, groups, weights = args
        if self.use_groups:
            logging.info("Group names are specified by column headers, group_names will be ignored!")
            keep_index = kwargs.get("keep_index", False)
            if keep_index:
                return DenseDataWithIndex(
                    background_data.values,
                    list(background_data.columns),
                    background_data.index.values,
                    background_data.index.name,
                    groups,
                    weights,
                )
            else:
                return DenseData(
                    background_data.values,
                    list(background_data.columns),
                    groups,
                    weights,
                )
        else:
            return background_data

    def explain(self,
                X: Union[np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.spmatrix],
                explanation_type: str = 'global',
                cat_vars_start_idx: Sequence[int] = None,
                cat_vars_enc_dim: Sequence[int] = None,
                **kwargs) -> Explanation:
        """
        Explains the instances in the array X.

        Parameters
        ----------
        X
            Array with instances to be explained.
        explanation_type
            Can be 'local' or 'global'. A string used to select the visualisations
            of the results. If global is selected, then stacked force plots and
            summary plots are displayed. If 'local' is specified but X has more
            than 1 row, the force plot for the first entry is displayed.
        cat_vars_start_idx
            A sequence containing the start indices of the categorical variables.
            If specified, cat_vars_enc_dim should also be specified.
        cat_vars_enc_dim
            A sequence containing the length of the encoding dimension for each
            categorical variable.
        kwargs
            Keyword arguments specifying explain behaviour. Valid arguments are:
                *nsamples: controls the number of predictor calls and therefore runtime.
                *l1_reg: controls the explanation sparsity.
            For more details, please see https://shap.readthedocs.io/en/latest/.
            Keyword arguments specifying display options:

        Returns
        -------
        explanation
            An explanation object containing the algorithm results.
        """

        # convert data to dense format if sparse
        if self.use_groups and isinstance(X, scipy.sparse.spmatrix):
            X = X.toarray()

        shap_values = self.explainer.shap_values(X, **kwargs)
        if cat_vars_start_idx is not None:
            self.summarise_result = True
            if self.use_groups:
                logger.warning(
                    "Specified both groups as well as summarisation for categorical variables. "
                    "By grouping, only one shap value is estimated for each categorical variable"
                    "so summarisation is not necessary."
                )
                self.summarise_result = False
            else:
                shap_values = sum_categories(shap_values, cat_vars_start_idx, cat_vars_enc_dim)
        expected_value = self.explainer.expected_value

        self.meta["explnations"][0] = explanation_type
        self.meta["params"].update([("summarised", self.summarise_result)])

        return self.build_explanation(X, shap_values, expected_value)

    def build_explanation(self, X: np.ndarray, shap_values: List[np.ndarray], expected_value: List) -> Explanation:
        """
        Create an explanation object.

        Parameters
        ----------
        X
            Array of instances to be explained.
        shap_values
            Each entry is a n_instances x n_features array, and the length of the list equals the dimensionality
            of the predictor output. The rows of each array correspond to the shap values for the instances with
            the corresponding row index in X
        expected_value
            A list containing the expected value of the prediction for each class.

        Returns
        -------
            An explanation containing a meta field with basic classifier metadata
        """

        data = {'shap_values': shap_values, 'expected_value': expected_value}

        return Explanation(meta=copy.deepcopy(self.meta), data=data)


def sum_categories(values: np.ndarray, start_idx: Sequence[int], enc_feat_dim: Sequence[int]):
    """
    For each entry in start_idx, the function sums the following k columns where k is the
    corresponding entry in the enc_feat_dim sequence. The columns whose indices are not in
    start_idx are left unchanged.

    Parameters
    ----------
    values
        The array whose columns will be summed.
    start_idx
        The start indices of the columns to be summed.
    enc_feat_dim
        The number of columns to be summed, one for each start index.

    Returns
    -------
    new_values
        An array whose columns have been summed according to the entries in start_idx and enc_feat_dim.
    """

    if start_idx is None or enc_feat_dim is None:
        raise ValueError("Both the start indices or the encoding dimension need to be specified!")

    if not len(enc_feat_dim) == len(start_idx):
        raise ValueError("The lengths of the sequences of start indices and encodings must be equal!")

    n_encoded_levels = sum(enc_feat_dim)
    if n_encoded_levels > values.shape[1]:
        raise ValueError("The sum of the encoded features dimensions exceeds data dimension!")

    new_values = np.zeros((values.shape[0]), values.shape[1] - n_encoded_levels + len(enc_feat_dim))
    enc_idx, new_vals_idx = 0, 0
    for idx in range(values.shape[1]):
        if idx in start_idx:
            feat_dim = start_idx[enc_idx]
            enc_idx += 1
            new_values[:, new_vals_idx] = np.sum(values[:, idx:idx + feat_dim], axis=1)
        else:
            new_values[:, new_vals_idx] = values[:, idx]
        new_vals_idx += 1

    return new_values

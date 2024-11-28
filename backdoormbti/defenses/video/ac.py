from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils.data import get_dataloader

from .base import InputFilteringBase


class ActivationClustering(InputFilteringBase):
    def __init__(self, args) -> None:
        super().__init__(args=args)
        self.args = args
        self.model_name = args.model

    def get_sanitized_lst(self, test_set=None):
        if test_set:
            self.dataset = test_set
        else:
            self.dataset = self.poison_train_set
        self.do_clustering()
        return self.is_clean_lst

    def do_clustering(self):
        num_classes = self.args.num_classes
        model = self.model
        for i, (x_batch, y_batch, *_) in enumerate(
            tqdm(self.poison_train_loader, desc="clustring")
        ):
            if not isinstance(x_batch, list):
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
            batch_activations = get_activations(
                self.model_name, model, x_batch, self.args
            )
            activation_dim = batch_activations.shape[-1]

            # initialize values list of lists on first run
            if i == 0:
                activations_by_class = [
                    np.empty((0, activation_dim)) for _ in range(num_classes)
                ]
                clusters_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]
                red_activations_by_class = [
                    np.empty((0, self.args.nb_dims)) for _ in range(num_classes)
                ]

            activations_by_class_i = segment_by_class(
                batch_activations, y_batch, num_classes
            )
            clusters_by_class_i, red_activations_by_class_i = cluster_activations(
                activations_by_class_i,
                nb_clusters=self.args.nb_clusters,
                nb_dims=self.args.nb_dims,
                reduce="PCA",
                clustering_method="KMeans",
            )

            for class_idx in range(num_classes):
                if activations_by_class_i[class_idx].shape[0] != 0:
                    activations_by_class[class_idx] = np.vstack(
                        [
                            activations_by_class[class_idx],
                            activations_by_class_i[class_idx],
                        ]
                    )
                    clusters_by_class[class_idx] = np.append(
                        clusters_by_class[class_idx], [clusters_by_class_i[class_idx]]
                    )
                    red_activations_by_class[class_idx] = np.vstack(
                        [
                            red_activations_by_class[class_idx],
                            red_activations_by_class_i[class_idx],
                        ]
                    )

        ### b. identify backdoor data according to classification results
        analyzer = ClusteringAnalyzer()
        if self.args.cluster_analysis == "smaller":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_size(clusters_by_class)
        elif self.args.cluster_analysis == "relative-size":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_relative_size(clusters_by_class)
        elif self.args.cluster_analysis == "distance":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_distance(
                clusters_by_class,
                separated_activations=red_activations_by_class,
            )
        elif self.args.cluster_analysis == "silhouette-scores":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_silhouette_score(
                clusters_by_class,
                reduced_activations_by_class=red_activations_by_class,
            )
        else:
            raise ValueError(
                "Unsupported cluster analysis technique " + self.args.cluster_analysis
            )

        batch_size = self.args.batch_size
        is_clean_lst = []
        # loop though the generator to generator a report
        last_loc = torch.zeros(self.args.num_classes).cpu().numpy().astype(int)
        for i, (x_batch, y_batch, *_) in enumerate(
            tqdm(self.poison_train_loader, desc="detecting")
        ):
            indices_by_class = segment_by_class(
                np.arange(batch_size), y_batch, self.args.num_classes
            )
            is_clean_lst_i = [0] * batch_size
            for class_idx, idxs in enumerate(indices_by_class):
                for idx_in_class, idx in enumerate(idxs):
                    is_clean_lst_i[idx] = assigned_clean_by_class[class_idx][
                        idx_in_class + last_loc[class_idx]
                    ]
                last_loc[class_idx] = last_loc[class_idx] + len(idxs)
            is_clean_lst += is_clean_lst_i
        # pop empty tail
        while len(is_clean_lst) > len(self.dataset):
            is_clean_lst.pop(-1)
        self.is_clean_lst = is_clean_lst


def segment_by_class(data, classes: np.ndarray, num_classes: int) -> List[np.ndarray]:
    try:
        width = data.size()[1]
        by_class: List[List[int]] = [[] for _ in range(num_classes)]

        for indx, feature in enumerate(classes):
            if len(classes.shape) == 2 and classes.shape[1] > 1:
                assigned = np.argmax(feature)
            else:
                assigned = int(feature)
            if torch.is_tensor(data[indx]):
                by_class[assigned].append(data[indx].cpu().numpy())
            else:
                by_class[assigned].append(data[indx])
        return [np.asarray(i, dtype=object).reshape(-1, width) for i in by_class]
    except:
        by_class: List[List[int]] = [[] for _ in range(num_classes)]

        for indx, feature in enumerate(classes):
            if len(classes.shape) == 2 and classes.shape[1] > 1:
                assigned = np.argmax(feature)

            else:
                assigned = int(feature)
            if torch.is_tensor(data[indx]):
                by_class[assigned].append(data[indx].cpu().numpy())
            else:
                by_class[assigned].append(data[indx])
        return [np.asarray(i, dtype=object) for i in by_class]


def cluster_activations(
    separated_activations: List[np.ndarray],
    nb_clusters: int = 2,
    nb_dims: int = 10,
    reduce: str = "FastICA",
    clustering_method: str = "KMeans",
    generator=None,
    clusterer_new=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clusters activations and returns two arrays.
    1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each data point
    in the class has been assigned.
    2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method.
    :param separated_activations: List where separated_activations[i] is a np matrix for the ith class where
           each row corresponds to activations for a given data point.
    :param nb_clusters: number of clusters (defaults to 2 for poison/clean).
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :param clustering_method: Clustering method to use, default is KMeans.
    :param generator: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations).
    :param clusterer_new: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations)
    """
    separated_clusters = []
    separated_reduced_activations = []

    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=nb_clusters, n_init="auto")
    else:
        raise ValueError(clustering_method + " clustering method not supported.")

    for activation in separated_activations:
        # Apply dimensionality reduction
        try:
            nb_activations = np.shape(activation)[1]
        except IndexError:
            activation = activation.reshape(1, -1)
            nb_activations = np.shape(activation)[1]
        if nb_activations > nb_dims & np.shape(activation)[0] > nb_dims:
            # TODO: address issue where if fewer samples than nb_dims this fails
            reduced_activations = reduce_dimensionality(
                activation, nb_dims=nb_dims, reduce=reduce
            )
        elif nb_activations <= nb_dims:
            reduced_activations = activation
        else:
            reduced_activations = activation[:, 0:(nb_dims)]
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        if (
            generator is not None
            and clusterer_new is not None
            and reduced_activations.shape[0] != 0
        ):
            clusterer_new = clusterer_new.partial_fit(reduced_activations)
            # NOTE: this may cause earlier predictions to be less accurate
            clusters = clusterer_new.predict(reduced_activations)
        elif reduced_activations.shape[0] != 1 and reduced_activations.shape[0] != 0:
            clusters = clusterer.fit_predict(reduced_activations)
        else:
            clusters = 1
        separated_clusters.append(clusters)

    return separated_clusters, separated_reduced_activations


def reduce_dimensionality(
    activations: np.ndarray, nb_dims: int = 10, reduce: str = "FastICA"
) -> np.ndarray:
    """
    Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.
    :param activations: Activations to be reduced.
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :return: Array with the reduced activations.
    """
    # pylint: disable=E0001
    from sklearn.decomposition import PCA, FastICA

    if reduce == "FastICA":
        projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == "PCA":
        projector = PCA(n_components=nb_dims)
    else:
        raise ValueError(reduce + " dimensionality reduction method not supported.")

    reduced_activations = projector.fit_transform(activations)
    return reduced_activations


def get_activations(name, model, x_batch, args):
    """get activations of the model for each sample
    name:
        the model name
    model:
        the train model
    x_batch:
        each batch for tain data
    """
    with torch.no_grad():
        model.eval()
        TOO_SMALL_ACTIVATIONS = 32
        assert name in [
            "preactresnet18",
            "vgg19",
            "vgg19_bn",
            "resnet18",
            "mobilenet_v3_large",
            "densenet161",
            "efficientnet_b3",
            "convnext_tiny",
            "vit_b_16",
            "audiocnn",
            "lstm",
            "xvector",
            "vggvox",
            "bert",
            "gpt2",
            "roberta",
        ]
        if name == "preactresnet18":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "vgg19":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch, return_hidden=True)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "vgg19_bn":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "resnet18":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.layer4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "mobilenet_v3_large":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "densenet161":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.features.register_forward_hook(layer_hook)
            _ = model(x_batch)
            outs[0] = torch.nn.functional.relu(outs[0])
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "efficientnet_b3":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "convnext_tiny":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.avgpool.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "vit_b_16":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                inps.append(inp[0].data)

            hook = model[1].heads.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = inps[0].view(inps[0].size(0), -1)
            hook.remove()
        elif name == "audiocnn":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                outs.append(out.data)

            hook = model.pool4.register_forward_hook(layer_hook)
            _ = model(x_batch)
            activations = outs[0].view(outs[0].size(0), -1)
            hook.remove()
        elif name == "lstm":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                inps.append(inp[0].data)

            hook = model.fc.register_forward_hook(layer_hook)
            waveform = x_batch.squeeze(dim=1)
            remain_size = waveform.numel() // (waveform.shape[0] * 40) * 40
            waveform = args.pre_trans(waveform[:, :remain_size])
            _ = model(waveform)
            activations = inps[0].view(inps[0].size(0), -1)
            hook.remove()
        elif name == "xvector":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                inps.append(inp[0].data)

            hook = model.output.register_forward_hook(layer_hook)
            waveform = x_batch.squeeze(dim=1)
            remain_size = waveform.numel() // (waveform.shape[0] * 40) * 40
            waveform = args.pre_trans(waveform[:, :remain_size])
            _ = model(waveform)
            activations = inps[0].view(inps[0].size(0), -1)
            hook.remove()
        elif name == "vggvox":
            inps, outs = [], []

            def layer_hook(module, inp, out):
                inps.append(inp[0].data)

            hook = model.fc.register_forward_hook(layer_hook)
            waveform = x_batch.squeeze(dim=1)
            remain_size = waveform.numel() // (waveform.shape[0] * 40) * 40
            _ = model(waveform)
            activations = inps[0].view(inps[0].size(0), -1)
            hook.remove()
        elif name in ["bert", "gpt2", "roberta"]:
            inputs = args.tokenizer(
                x_batch, padding=True, truncation=True, return_tensors="pt"
            )
            inputs.to(args.device)
            outs = model(**inputs)
            activations = outs.hidden_states[-1][:, 0, :]

    return activations


class ClusteringAnalyzer:
    """
    Class for all methodologies implemented to analyze clusters and determine whether they are poisonous.
    """

    @staticmethod
    def assign_class(
        clusters: np.ndarray, clean_clusters: List[int], poison_clusters: List[int]
    ) -> np.ndarray:
        """
        Determines whether each data point in the class is in a clean or poisonous cluster
        :param clusters: `clusters[i]` indicates which cluster the i'th data point is in.
        :param clean_clusters: List containing the clusters designated as clean.
        :param poison_clusters: List containing the clusters designated as poisonous.
        :return: assigned_clean: `assigned_clean[i]` is a boolean indicating whether the ith data point is clean.
        """

        assigned_clean = np.empty(np.shape(clusters))
        assigned_clean[np.isin(clusters, clean_clusters)] = 1
        assigned_clean[np.isin(clusters, poison_clusters)] = 0
        return assigned_clean

    def analyze_by_size(
        self, separated_clusters: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:
        """
        Designates as poisonous the cluster with less number of items on it.
        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class.
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether
                 a given data point was determined to be clean (as opposed to poisonous) and
                 summary_poison_clusters: array, where summary_poison_clusters[i][j]=1 if cluster j of class i was
                 classified as poison, otherwise 0
                 report: Dictionary with summary of the analysis
        """
        report: Dict[str, Any] = {
            "cluster_analysis": "smaller",
            "suspicious_clusters": 0,
        }

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [
            [0 for _ in range(nb_clusters)] for _ in range(nb_classes)
        ]

        for i, clusters in enumerate(separated_clusters):
            # assume that smallest cluster is poisonous and all others are clean
            sizes = np.bincount(clusters)
            total_dp_in_class = np.sum(sizes)
            poison_clusters: List[int] = [int(np.argmin(sizes))]
            clean_clusters = list(set(clusters) - set(poison_clusters))
            for p_id in poison_clusters:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters:
                summary_poison_clusters[i][c_id] = 0

            assigned_clean = self.assign_class(
                clusters, clean_clusters, poison_clusters
            )
            all_assigned_clean.append(assigned_clean)
            # Generate report for this class:
            report_class = dict()
            for cluster_id in range(nb_clusters):
                ptc = sizes[cluster_id] / total_dp_in_class
                susp = cluster_id in poison_clusters
                dict_i = dict(
                    ptc_data_in_cluster=round(ptc, 2), suspicious_cluster=susp
                )
                dict_cluster: Dict[str, Dict[str, int]] = {
                    "cluster_" + str(cluster_id): dict_i
                }
                report_class.update(dict_cluster)

            report["Class_" + str(i)] = report_class

        report["suspicious_clusters"] = (
            report["suspicious_clusters"] + np.sum(summary_poison_clusters).item()
        )
        return (
            np.asarray(all_assigned_clean, dtype=object),
            summary_poison_clusters,
            report,
        )

    def analyze_by_distance(
        self,
        separated_clusters: List[np.ndarray],
        separated_activations: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:
        """
        Assigns a cluster as poisonous if its median activation is closer to the median activation for another class
        than it is to the median activation of its own class. Currently, this function assumes there are only two
        clusters per class.
        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class.
        :param separated_activations: list where separated_activations[i] is a 1D array of [0,1] for [poison,clean].
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined
                 to be clean (as opposed to poisonous) and summary_poison_clusters: array, where
                 summary_poison_clusters[i][j]=1 if cluster j of class i was classified as poison, otherwise 0
                 report: Dictionary with summary of the analysis.
        """

        report: Dict[str, Any] = {"cluster_analysis": 0.0}
        all_assigned_clean = []
        cluster_centers = []

        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [
            [0 for _ in range(nb_clusters)] for _ in range(nb_classes)
        ]

        # assign centers
        for _, activations in enumerate(separated_activations):
            cluster_centers.append(np.median(activations, axis=0))

        for i, (clusters, activation) in enumerate(
            zip(separated_clusters, separated_activations)
        ):
            clusters = np.array(clusters)
            cluster0_center = np.median(activation[np.where(clusters == 0)], axis=0)
            cluster1_center = np.median(activation[np.where(clusters == 1)], axis=0)

            cluster0_distance = np.linalg.norm(cluster0_center - cluster_centers[i])
            cluster1_distance = np.linalg.norm(cluster1_center - cluster_centers[i])

            cluster0_is_poison = False
            cluster1_is_poison = False

            dict_k = dict()
            dict_cluster_0 = dict(cluster0_distance_to_its_class=str(cluster0_distance))
            dict_cluster_1 = dict(cluster1_distance_to_its_class=str(cluster1_distance))
            for k, center in enumerate(cluster_centers):
                if k == i:
                    pass
                else:
                    cluster0_distance_to_k = np.linalg.norm(cluster0_center - center)
                    cluster1_distance_to_k = np.linalg.norm(cluster1_center - center)
                    if (
                        cluster0_distance_to_k < cluster0_distance
                        and cluster1_distance_to_k > cluster1_distance
                    ):
                        cluster0_is_poison = True
                    if (
                        cluster1_distance_to_k < cluster1_distance
                        and cluster0_distance_to_k > cluster0_distance
                    ):
                        cluster1_is_poison = True

                    dict_cluster_0["distance_to_class_" + str(k)] = str(
                        cluster0_distance_to_k
                    )
                    dict_cluster_0["suspicious"] = str(cluster0_is_poison)
                    dict_cluster_1["distance_to_class_" + str(k)] = str(
                        cluster1_distance_to_k
                    )
                    dict_cluster_1["suspicious"] = str(cluster1_is_poison)
                    dict_k.update(dict_cluster_0)
                    dict_k.update(dict_cluster_1)

            report_class = dict(cluster_0=dict_cluster_0, cluster_1=dict_cluster_1)
            report["Class_" + str(i)] = report_class

            poison_clusters = []
            if cluster0_is_poison:
                poison_clusters.append(0)
                summary_poison_clusters[i][0] = 1
            else:
                summary_poison_clusters[i][0] = 0

            if cluster1_is_poison:
                poison_clusters.append(1)
                summary_poison_clusters[i][1] = 1
            else:
                summary_poison_clusters[i][1] = 0

            clean_clusters = list(set(clusters) - set(poison_clusters))
            assigned_clean = self.assign_class(
                clusters, clean_clusters, poison_clusters
            )
            all_assigned_clean.append(assigned_clean)

        all_assigned_clean = np.asarray(all_assigned_clean, dtype=object)
        return all_assigned_clean, summary_poison_clusters, report

    def analyze_by_relative_size(
        self,
        separated_clusters: List[np.ndarray],
        size_threshold: float = 0.35,
        r_size: int = 2,
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:
        """
        Assigns a cluster as poisonous if the smaller one contains less than threshold of the data.
        This method assumes only 2 clusters
        :param separated_clusters: List where `separated_clusters[i]` is the cluster assignments for the ith class.
        :param size_threshold: Threshold used to define when a cluster is substantially smaller.
        :param r_size: Round number used for size rate comparisons.
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined
                 to be clean (as opposed to poisonous) and summary_poison_clusters: array, where
                 summary_poison_clusters[i][j]=1 if cluster j of class i was classified as poison, otherwise 0
                 report: Dictionary with summary of the analysis.
        """

        size_threshold = round(size_threshold, r_size)
        report: Dict[str, Any] = {
            "cluster_analysis": "relative_size",
            "suspicious_clusters": 0,
            "size_threshold": size_threshold,
        }

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [
            [0 for _ in range(nb_clusters)] for _ in range(nb_classes)
        ]

        for i, clusters in enumerate(separated_clusters):
            sizes = np.bincount(clusters)
            total_dp_in_class = np.sum(sizes)

            if np.size(sizes) > 2:
                raise ValueError(
                    " RelativeSizeAnalyzer does not support more than two clusters."
                )
            percentages = np.round(sizes / float(np.sum(sizes)), r_size)
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)

            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0

            assigned_clean = self.assign_class(
                clusters, clean_clusters, poison_clusters
            )
            all_assigned_clean.append(assigned_clean)

            # Generate report for this class:
            report_class = dict()
            for cluster_id in range(nb_clusters):
                ptc = sizes[cluster_id] / total_dp_in_class
                susp = cluster_id in poison_clusters
                dict_i = dict(
                    ptc_data_in_cluster=round(ptc, 2), suspicious_cluster=susp
                )

                dict_cluster = {"cluster_" + str(cluster_id): dict_i}
                report_class.update(dict_cluster)

            report["Class_" + str(i)] = report_class

        report["suspicious_clusters"] = (
            report["suspicious_clusters"] + np.sum(summary_poison_clusters).item()
        )
        return (
            np.asarray(all_assigned_clean, dtype=object),
            summary_poison_clusters,
            report,
        )

    def analyze_by_silhouette_score(
        self,
        separated_clusters: list,
        reduced_activations_by_class: list,
        size_threshold: float = 0.35,
        silhouette_threshold: float = 0.1,
        r_size: int = 2,
        r_silhouette: int = 4,
    ) -> Tuple[np.ndarray, List[List[int]], Dict[str, int]]:
        """
        Analyzes clusters to determine level of suspiciousness of poison based on the cluster's relative size
        and silhouette score.
        Computes a silhouette score for each class to determine how cohesive resulting clusters are.
        A low silhouette score indicates that the clustering does not fit the data well, and the class can be considered
        to be un-poison. Conversely, a high silhouette score indicates that the clusters reflect true splits in the
        data.
        The method concludes that a cluster is poison based on the silhouette score and the cluster relative size.
        If the relative size is too small, below a size_threshold and at the same time
        the silhouette score is higher than silhouette_threshold, the cluster is classified as poisonous.
        If the above thresholds are not provided, the default ones will be used.
        :param separated_clusters: list where `separated_clusters[i]` is the cluster assignments for the ith class.
        :param reduced_activations_by_class: list where separated_activations[i] is a 1D array of [0,1] for
               [poison,clean].
        :param size_threshold: (optional) threshold used to define when a cluster is substantially smaller. A default
        value is used if the parameter is not provided.
        :param silhouette_threshold: (optional) threshold used to define when a cluster is cohesive. Default
        value is used if the parameter is not provided.
        :param r_size: Round number used for size rate comparisons.
        :param r_silhouette: Round number used for silhouette rate comparisons.
        :return: all_assigned_clean, summary_poison_clusters, report:
                 where all_assigned_clean[i] is a 1D boolean array indicating whether a given data point was determined
                 to be clean (as opposed to poisonous) summary_poison_clusters: array, where
                 summary_poison_clusters[i][j]=1 if cluster j of class j was classified as poison
                 report: Dictionary with summary of the analysis.
        """

        # pylint: disable=E0001
        from sklearn.metrics import silhouette_score

        size_threshold = round(size_threshold, r_size)
        silhouette_threshold = round(silhouette_threshold, r_silhouette)
        report: Dict[str, Any] = {
            "cluster_analysis": "silhouette_score",
            "size_threshold": str(size_threshold),
            "silhouette_threshold": str(silhouette_threshold),
        }

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters: List[List[int]] = [
            [0 for _ in range(nb_clusters)] for _ in range(nb_classes)
        ]

        for i, (clusters, activations) in enumerate(
            zip(separated_clusters, reduced_activations_by_class)
        ):
            bins = np.bincount(clusters)
            if np.size(bins) > 2:
                raise ValueError("Analyzer does not support more than two clusters.")

            percentages = np.round(bins / float(np.sum(bins)), r_size)
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)

            # Generate report for class
            silhouette_avg = round(
                silhouette_score(activations, clusters), r_silhouette
            )
            dict_i: Dict[str, Any] = dict(
                sizes_clusters=str(bins),
                ptc_cluster=str(percentages),
                avg_silhouette_score=str(silhouette_avg),
            )

            if np.shape(poison_clusters)[1] != 0:
                # Relative size of the clusters is suspicious
                if silhouette_avg > silhouette_threshold:
                    # In this case the cluster is considered poisonous
                    clean_clusters = np.where(percentages < size_threshold)
                    dict_i.update(suspicious=True)
                else:
                    poison_clusters = [[]]
                    clean_clusters = np.where(percentages >= 0)
                    dict_i.update(suspicious=False)
            else:
                # If relative size of the clusters is Not suspicious, we conclude it's not suspicious.

                dict_i.update(suspicious=False)

            report_class: Dict[str, Dict[str, bool]] = {"class_" + str(i): dict_i}
            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1

            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0

            assigned_clean = self.assign_class(
                clusters, clean_clusters, poison_clusters
            )
            all_assigned_clean.append(assigned_clean)
            report.update(report_class)

        return (
            np.asarray(all_assigned_clean, dtype=object),
            summary_poison_clusters,
            report,
        )

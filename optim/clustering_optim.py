import numpy as np
import pandas as pd
import os
import pickle
import json
from sentence_transformers import SentenceTransformer
import nni
import wandb
import datetime as dt
import cuml
import hdbscan
from typing import Dict
import sklearn.metrics as mt

MLFLOW_EXPERIMENT_NAME='CLUSTERING_tests_en_1'
MLFLOW_URI='/mml/mlflow_orlica_train'
NNI_DISABLED = 'STANDALONE'
RANDOM_STATE = None if nni.get_experiment_id() == NNI_DISABLED else 123
TEXT_DATAFILE = "/mnt/workdata/_WORK_/customer_intent_clustering/data/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"
TEXT_FIELDNAME = "utterance"
MODEL_PATH = '/mnt/Data2/pretrained_models/nlp_models/sentence/'
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDINGS_PATH = "/mnt/workdata/_WORK_/customer_intent_clustering/temp/"
LIMIT_DATA_FRACTION=1.0
WANDB_EXPERIMENT = "CIC_working"
def build_embeddings_dataset(
        text_datafile: str,
        text_fieldname: str,
        model_path: str,
        model_filename: str,
        embeddings_output_filename: str,
        limit_to_fraction=1.0) -> np.ndarray:
    """
    Check if embeddings file exists and creates one if it doesn't.
    If embeddings file is present loads it.

    :param text_datafile: filename of source dataframe with text column to create embeddings for,
    :param text_fieldname: the name of the text column,
    :param model_path: the path to model used for embeddings computation,
    :param model_filename: the directory with mdoel files or model file
    :param embeddings_output_filename: the name of output file to store embeddings in
    :return: array of embeddings
    """
    if not os.path.isfile(embeddings_output_filename):
        df = pd.read_csv(text_datafile)
        if limit_to_fraction<1.0:
            df=df.sample(frac=limit_to_fraction, axis=0).reset_index(drop=True)
        print(
            f"The dataset contains {df.shape[0]} short texts, assigned to {df['intent'].nunique()} intents, grouped into {df['category'].nunique()} categories")
        model = SentenceTransformer(
            model_name_or_path=os.path.join(model_path, model_filename)
        )
        embeds = model.encode(
            sentences=df[text_fieldname],
            batch_size=128,
            show_progress_bar=False,
            output_value='sentence_embedding',
            convert_to_numpy=True,
            device='cuda',
            normalize_embeddings=True
        )
        with open(embeddings_output_filename, "wb") as fp:
            pickle.dump(embeds, fp)
        print(f"Embeddings stored as {embeddings_output_filename}")
    else:
        with open(embeddings_output_filename, "rb") as fp:
            embeds = pickle.load(fp)
        print(f"Embeddings loaded from {embeddings_output_filename}")
    return embeds

def define_params(nni_mode: str):
    params = {
        # umap parameters
        "umap_n_components": 40,
        "umap_n_neighbors": 25,
        "umap_min_distance": 0.25,
        "umap_spread": 1.0,
        "umap_learning_rate": 0.1,
        "umap_init": 'spectral',
        "umap_random_state": RANDOM_STATE,
        # hdbscan parameters
        "hdbs_min_cluster_size": 20,
        "hdbs_min_samples": 20,
        "hdbs_max_cluster_size": 500,
        "hdbs_cluster_selection_epsilon": 0.,
        "hdbs_p": 3,
        "hdbs_cluster_selection_method": 'eom',
        "hdbs_metric": "euclidean",
        "hdbs_alpha": 1.0
    }

    if nni_mode != NNI_DISABLED:
        nni_params = nni.get_next_parameters()
        print(json.dumps(nni_params, indent=3))

        # decompose spaces for variants of parameters
        decomposed= {}
        for k in nni_params.keys():
            if k.startswith('_'):
                for k_ in nni_params[k].keys():
                    decomposed[k_] = nni_params[k][k_]
        nni_params = {k: nni_params[k] for k in nni_params.keys() if k.startswith("_")==False}
        if len(decomposed)>0:
            nni_params = {**nni_params, **decomposed}

        # correct typws of params provided by nni
        nni_params = {k: type(params[k])(nni_params[k]) for k in nni_params.keys() if params[k] is not None}

        for k in params.keys():
            # add params not provided from nni
            if k not in nni_params.keys():
                nni_params[k] = params[k]
        return nni_params
    else:
        return params

def create_umap_projection(
    embedding_data: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_distance: float,
    spread: float,
    learning_rate: float,
    init: str,
    random_state: int,
    ):
    mapper = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='cosine',
        learning_rate=learning_rate,
        min_dist = min_distance,
        init=init,
        spread = spread,
        random_state=random_state,
        output_type="numpy")
    projection = mapper.fit_transform(X = embedding_data, convert_dtype=True)
    return mapper, projection

def find_clusters(
        projection_data: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
        max_cluster_size: int,
        cluster_selection_epsilon: float,
        p:int,
        cluster_selection_method,
        metric: str,
        alpha: float
):
    clusterer = cuml.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        alpha = alpha,
        p=p,
        cluster_selection_method=cluster_selection_method,
        output_type='numpy')
    clusters_labels = clusterer.fit_predict(projection_data)
    return clusterer, clusters_labels


def assess_clustering_results(clusters_labels,
                              clustered_data: np.ndarray,
                              params: Dict) -> Dict:
    results = {}
    results['validity_index'] = hdbscan.validity_index(
        clustered_data.astype('float64'), labels=cluster_labels, metric=params['hdbs_metric'])
    results["calinski_harabasz_score"] = mt.calinski_harabasz_score(X=clustered_data, labels=clusters_labels)
    results["davies_bouldin_score"]= mt.davies_bouldin_score(X=clustered_data, labels=clusters_labels)
    return results


if __name__ == '__main__':
    NNI_EXP_ID = nni.get_experiment_id()
    NNI_RUN_ID = nni.get_trial_id()
    NNI_SEQ_ID = nni.get_sequence_id()

    if NNI_EXP_ID==NNI_DISABLED:
        WANDB_RUN_NAME = f"CIC_{dt.datetime.now():%m%d%H%M%S}"
    else:
        WANDB_RUN_NAME = NNI_RUN_ID
    EMBEDDINGS_DATA_FILENAME = os.path.join(EMBEDDINGS_PATH, f"embeds_{NNI_EXP_ID}_{MODEL_NAME}.pkl")

    params = define_params(NNI_EXP_ID)
    print("Parameters set:")
    print(json.dumps(params, indent=3))

    run_config = {**params,
                  **{'model': MODEL_NAME}}

    with wandb.init(project=WANDB_EXPERIMENT, name=WANDB_RUN_NAME, save_code=False,
                    config =run_config) as run:

        # create embeddings
        embeds = build_embeddings_dataset(
            text_datafile=TEXT_DATAFILE,
            text_fieldname=TEXT_FIELDNAME,
            model_path=MODEL_PATH,
            model_filename=MODEL_NAME,
            embeddings_output_filename=EMBEDDINGS_DATA_FILENAME,
            limit_to_fraction=LIMIT_DATA_FRACTION)
        print(f'Embeddings data dimensionality: {embeds.shape}')

        # create projection
        proj_params = {k[len('umap_'):]: params[k] for k in params.keys() if k.startswith('umap_')}
        projector, projection_data = create_umap_projection(embedding_data=embeds, **proj_params)
        print("projection parameters:")
        print(json.dumps(proj_params, indent=3))
        print(f'Projection data dimensionality: {projection_data.shape}')

        # perform clustering
        clu_params = {k[len('hdbs_'):]:params[k] for k in params.keys() if k.startswith('hdbs_')}
        clusterer, cluster_labels = find_clusters(projection_data=projection_data, **clu_params)
        print("clustering parameters:")
        print(json.dumps(clu_params, indent=3))
        print(f'Projection data dimensionality: {projection_data.shape}')

        # compute metrics
        clu_metrics =assess_clustering_results(clusters_labels=cluster_labels, clustered_data=projection_data,
                                  params=params)

        # finalize results dict
        results = {
            "trial_number": NNI_SEQ_ID,
            "n_clusters": len(set([x for x in cluster_labels if x>=0])),
            "ooc_count": len([x for x in cluster_labels if x<0]),
            "ooc_fraction": len([x for x in cluster_labels if x<0])/len(cluster_labels),
        }
        results = {**clu_metrics, **results}
        if NNI_EXP_ID != NNI_DISABLED:
            nni_results={**{'default': results['validity_index']}}
            nni.report_final_result(nni_results)
        wandb.log(results)







import numpy as np
from src.features.guidance_analysis import grouped_probe, mixed_ok_ratio


def test_probe_keeps_duplicate_groups_in_one_fold():
    rng=np.random.default_rng(4)
    groups=np.repeat(np.arange(20).astype(str),2)
    targets=np.repeat(np.arange(20)%2,2)
    vectors=rng.normal(size=(40,6))+targets[:,None]
    probabilities,assignments,evidence=grouped_probe(vectors,targets,groups)
    assert np.isfinite(probabilities).all()
    assert all(len(set(assignments[groups==group]))==1 for group in groups)
    assert all(not set(fold['train_groups']) & set(fold['validation_groups']) for fold in evidence)


def test_knn_excludes_all_same_image_duplicates():
    queries=np.array([[1.,0.]])
    pool=np.array([[1.,0.],[1.,0.],[.9,.1],[.8,.2]])
    ratio=mixed_ok_ratio(queries,pool,np.array([1,1,0,0]),np.array(['a']),np.array(['a','a','b','c']),k=2)
    assert ratio[0]==1.

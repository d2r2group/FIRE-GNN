"""Implementation based on the template of ALIGNN."""

import imp
import random
from pathlib import Path
from typing import Optional
# from typing import Dict, List, Optional, Set, Tuple
import os
import torch
import ast
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms,  pmg_to_atoms
import structlog
from pymatgen.core import Structure
from comformer.graphs import PygGraph, PygStructureDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
# from sklearn.pipeline import Pipeline
import pickle as pk
from sklearn.preprocessing import StandardScaler
# use pandas progress_apply
tqdm.pandas()
logger = structlog.get_logger()
import hashlib


def load_dataset(
    name: str = "D2R2_surface_data", 
    data_path: str = "data/DFT_data_augmented.csv",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    print(f"DEBUG: In load_dataset, target='{target}' (type: {type(target)})")
    logger.info(f"reading data from path {data_path}")
    logger.info(f"limit parameter value: {limit}")
    logger.info(f"The target property is {target}")  # Log target property immediately
    
    df = pd.read_csv(data_path, on_bad_lines="skip")
    if limit is not None:
        df = df[:limit]
    logger.info(f"Data is loaded from {data_path}")
    
    # Check if 'flipped' column exists and use it for jid if present
    if 'flipped' in df.columns:
        logger.info("Using 'flipped' column in jid construction")
        df["jid"] = df["mpid"].astype(str) + df["miller"].astype(str) + df["term"].astype(str) + df['flipped'].astype(str)
    else:
        logger.info("'flipped' column not found, using original jid construction")
        df["jid"] = df["mpid"].astype(str) + df["miller"].astype(str) + df["term"].astype(str)
    
    # For multi-property training: if target=="all" and dataset is D2R2_surface_data,
    # combine WF_bottom, WF_top, and cleavage_energy into one list.
    if target == "all":
        logger.info("Combining WF_bottom, WF_top, and cleavage_energy into 'all' field for D2R2_surface_data")
        df["all"] = df.apply(
            lambda x: [x["WF_bottom"], x["WF_top"], x["cleavage_energy"]],
            axis=1
        )
    # For training only work function properties, combine WF_bottom and WF_top
    elif target == "WF":
        logger.info("Combining WF_bottom and WF_top into 'WF' field for D2R2_surface_data")
        df["WF"] = df.apply(
            lambda x: [x["WF_bottom"], x["WF_top"]],
            axis=1
        )
    elif target is not None:
        # Ensure the target property exists in the dataframe
        if target in df.columns:
            logger.info(f"Found target property '{target}' in the dataset with {df[target].count()} non-null values")
        else:
            logger.error(f"Target property '{target}' not found in the dataset. Available columns: {list(df.columns)}")
            raise ValueError(f"Target property '{target}' not found in dataset")

    if "slab" in df.columns:
        df = df.rename(columns={"slab": "atoms"})
    
    logger.info(f"There are {len(df)} rows in this df")
    return df


def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def get_cache_key(jid, neighbor_strategy, cutoff, max_neighbors, use_canonize, use_lattice, use_angle):
    """Generate cache key from jid and graph parameters."""
    params = f"{jid}_{neighbor_strategy}_{cutoff}_{max_neighbors}_{use_canonize}_{use_lattice}_{use_angle}"
    return hashlib.md5(params.encode()).hexdigest()


def load_graphs_to_memory(df, neighbor_strategy, cutoff, max_neighbors, use_canonize, use_lattice, use_angle, cache_dir):
    """Load all available graphs from disk cache to memory."""
    memory_cache = {}
    cache_hits = 0
    
    for jid in df['jid'].values:
        cache_key = get_cache_key(jid, neighbor_strategy, cutoff, max_neighbors, use_canonize, use_lattice, use_angle)
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                memory_cache[cache_key] = torch.load(cache_file, map_location='cpu')
                cache_hits += 1
            except Exception as e:
                logger.warning(f"Failed to load cache for {jid}: {e}")
    
    logger.info(f"Loaded {cache_hits}/{len(df)} graphs from disk cache")
    return memory_cache


def atoms_to_graph_with_cache(atoms, jid, memory_cache, neighbor_strategy, cutoff, max_neighbors, use_canonize, use_lattice, use_angle, cache_dir):
    """Convert structure dict to graph with caching."""
    cache_key = get_cache_key(jid, neighbor_strategy, cutoff, max_neighbors, use_canonize, use_lattice, use_angle)
    
    if cache_key in memory_cache:
        return memory_cache[cache_key]
    
    # Compute new graph
    structure = pmg_to_atoms(Structure.from_dict(eval(atoms)))
    graph = PygGraph.atom_dgl_multigraph(
        structure,
        neighbor_strategy=neighbor_strategy,
        cutoff=cutoff,
        atom_features="atomic_number",
        max_neighbors=max_neighbors,
        compute_line_graph=False,
        use_canonize=use_canonize,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )
    
    # Save to disk cache for future use
    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        torch.save(graph, cache_file)
    except Exception as e:
        logger.warning(f"Failed to save cache for {jid}: {e}")
    
    return graph


def load_pyg_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    use_lattice: bool = False,
    use_angle: bool = False,
):
    """Construct crystal graphs with disk caching.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path("./cache/graphs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading graphs from cache and computing new ones...")
    
    # Load cached graphs to memory
    memory_cache = load_graphs_to_memory(df, neighbor_strategy, cutoff, max_neighbors, use_canonize, use_lattice, use_angle, cache_dir)
    
    # Process graphs with cache
    graphs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing graphs"):
        graph = atoms_to_graph_with_cache(
            row["atoms"], row["jid"], memory_cache, 
            neighbor_strategy, cutoff, max_neighbors, 
            use_canonize, use_lattice, use_angle, cache_dir
        )
        graphs.append(graph)
    
    # Clear memory cache
    del memory_cache
    logger.info(f"Processed {len(graphs)} graphs")
    
    return graphs


def get_id_train_val_test_from_splits(dat, id_tag="jid"):
    """Get train, val, test IDs from predetermined split column."""
    try:
        id_train = []
        id_val = []
        id_test = []
        
        for idx, item in enumerate(dat):
            split_value = item.get('split', '').lower().strip()
            if split_value in ['train', 'training']:
                id_train.append(idx)
            elif split_value in ['val', 'validation', 'valid']:
                id_val.append(idx)
            elif split_value in ['test', 'testing']:
                id_test.append(idx)
            else:
                logger.warning(f"Unknown split value '{split_value}' for item {item.get(id_tag, idx)}. Skipping.")
        
        # Validate that we have data in each split
        if len(id_train) == 0:
            raise ValueError("No training samples found in split column")
        if len(id_val) == 0:
            raise ValueError("No validation samples found in split column")
        if len(id_test) == 0:
            raise ValueError("No test samples found in split column")
            
        logger.info(f"Using predetermined splits: train={len(id_train)}, val={len(id_val)}, test={len(id_test)}")
        return id_train, id_val, id_test
        
    except Exception as e:
        logger.error(f"Error processing predetermined splits: {e}")
        raise


def get_id_train_val_test(
    total_size=None,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_pyg_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    use_lattice=False,
    use_angle=False,
    data_from='Jarvis',
    use_save=False,
    mean_train=None,
    std_train=None,
    now=False, # for test
):
    """Get pyg Dataset."""
    df = pd.DataFrame(dataset)
    
    # Modify to handle multi-output case
    if target == "all":
        vals = np.array([x for x in df[target].values])
        output_features = 3  # Number of properties we're predicting
    elif target == "WF":
        vals = np.array([x for x in df[target].values])
        output_features = 2  # Number of properties we're predicting (WF_bottom, WF_top)
    else:
        vals = df[target].values
        output_features = 1
    
    output_dir = "./saved_data/" + tmp_name + "test_graph_angle.pkl" # for fast test use
    print("data range", np.max(vals), np.min(vals))
    print(output_dir)
    print('graphs not saved')
    graphs = load_pyg_graphs(
        df,
        name=name,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
        use_angle=use_angle,
    )
    if mean_train == None:
        # SCALING DISABLED: To re-enable normalization, uncomment below:
        # mean_train = np.mean(vals)
        # std_train = np.std(vals)
        
        mean_train = 0.0  # Hardcoded to disable normalization
        std_train = 1.0   # Hardcoded to disable normalization
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    else:
        data = PygStructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            neighbor_strategy=neighbor_strategy,
            mean_train=mean_train,
            std_train=std_train,
        )
    return data, mean_train, std_train


def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=[],
    target: str = "formation_energy_peratom",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    use_predetermined_splits=True,
    output_features=1,
    output_dir=None,
    matrix_input=False,
    pyg_input=False,
    use_lattice=False,
    use_angle=False,
    use_save=True,
    mp_id_list=None,
    data_path=None,
):
    """Help function to set up JARVIS train and val dataloaders."""
    # Log important parameters
    logger.info(f"Setting up data loaders for dataset: {dataset}")
    logger.info(f"Target property: {target}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output features: {output_features}")
    # data loading
    mean_train=None
    std_train=None
    assert (matrix_input and pyg_input) == False
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not dataset_array:
        # First load the dataset
        print(f"DEBUG: Calling load_dataset with target='{target}'")
        df = load_dataset(name=dataset, data_path=data_path, target=target)
        
        # Make sure the 'all' field is created if target is 'all'
        if target == "all" and "all" not in df.columns:
            print("Creating 'all' field from WF_bottom, WF_top, and cleavage_energy columns")
            required_cols = ["WF_bottom", "WF_top", "cleavage_energy"]
            if all(col in df.columns for col in required_cols):
                df["all"] = df.apply(
                    lambda x: [x["WF_bottom"], x["WF_top"], x["cleavage_energy"]],
                    axis=1
                )
            else:
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Cannot create 'all' field. Missing columns: {missing}")
        # Make sure the 'WF' field is created if target is 'WF'
        elif target == "WF" and "WF" not in df.columns:
            print("Creating 'WF' field from WF_bottom and WF_top columns")
            required_cols = ["WF_bottom", "WF_top"]
            if all(col in df.columns for col in required_cols):
                df["WF"] = df.apply(
                    lambda x: [x["WF_bottom"], x["WF_top"]],
                    axis=1
                )
            else:
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Cannot create 'WF' field. Missing columns: {missing}")
        
        d = df.to_dict(orient="records")
    else:
        d = dataset_array
        # for ii, i in enumerate(pc_y):
        #    d[ii][target] = pc_y[ii].tolist()

    dat = []
    if classification_threshold is not None:
        print(
            "Using ",
            classification_threshold,
            " for classifying ",
            target,
            " data.",
        )
        print("Converting target data into 1 and 0.")
    all_targets = []

    # TODO:make an all key in qm9_dgl
    if dataset == "qm9_dgl" and target == "all":
        print("Making all qm9_dgl")
        tmp = []
        for ii in d:
            ii["all"] = [
                ii["mu"],
                ii["alpha"],
                ii["homo"],
                ii["lumo"],
                ii["gap"],
                ii["r2"],
                ii["zpve"],
                ii["U0"],
                ii["U"],
                ii["H"],
                ii["G"],
                ii["Cv"],
            ]
            tmp.append(ii)
        print("Made all qm9_dgl")
        d = tmp
    # logger.info(d)
    for i in d:
        # If target is 'all' or 'WF' but not present in the data, create it on the fly
        if target == "all" and target not in i:
            required_cols = ["WF_bottom", "WF_top", "cleavage_energy"]
            if all(col in i for col in required_cols):
                i[target] = [i["WF_bottom"], i["WF_top"], i["cleavage_energy"]]
                print(f"Creating 'all' field on the fly: {i[target]}")
            else:
                missing = [col for col in required_cols if col not in i]
                print(f"Warning: Cannot create 'all' field. Missing columns: {missing}. Available: {list(i.keys())}")
                continue  # Skip this item
        elif target == "WF" and target not in i:
            required_cols = ["WF_bottom", "WF_top"]
            if all(col in i for col in required_cols):
                i[target] = [i["WF_bottom"], i["WF_top"]]
                print(f"Creating 'WF' field on the fly: {i[target]}")
            else:
                missing = [col for col in required_cols if col not in i]
                print(f"Warning: Cannot create 'WF' field. Missing columns: {missing}. Available: {list(i.keys())}")
                continue  # Skip this item
        
        try:
            if isinstance(i[target], list):  # multioutput target
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)
            elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
            ):
                if target_multiplication_factor is not None:
                    i[target] = i[target] * target_multiplication_factor
                if classification_threshold is not None:
                    if i[target] <= classification_threshold:
                        i[target] = 0
                    elif i[target] > classification_threshold:
                        i[target] = 1
                    else:
                        raise ValueError(
                            "Check classification data type.",
                            i[target],
                            type(i[target]),
                        )
                dat.append(i)
                all_targets.append(i[target])
        except KeyError:
            print(f"Warning: Target '{target}' not found in data. Keys: {list(i.keys())}")
            continue  # Skip this item
    
    if mp_id_list is not None:
        if mp_id_list == 'bulk':
            print('using mp bulk dataset')
            with open('/data/keqiangyan/bulk_shear/bulk_megnet_train.pkl', 'rb') as f:
                dataset_train = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/bulk_megnet_val.pkl', 'rb') as f:
                dataset_val = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/bulk_megnet_test.pkl', 'rb') as f:
                dataset_test = pk.load(f)
        
        if mp_id_list == 'shear':
            print('using mp shear dataset')
            with open('/data/keqiangyan/bulk_shear/shear_megnet_train.pkl', 'rb') as f:
                dataset_train = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/shear_megnet_val.pkl', 'rb') as f:
                dataset_val = pk.load(f)
            with open('/data/keqiangyan/bulk_shear/shear_megnet_test.pkl', 'rb') as f:
                dataset_test = pk.load(f)

    else:
        # Check if we should use predetermined splits
        use_split_column = False
        if use_predetermined_splits and len(dat) > 0:
            # Check if split column exists in the data
            if 'split' in dat[0]:
                logger.info("Split column detected. Attempting to use predetermined splits.")
                try:
                    id_train, id_val, id_test = get_id_train_val_test_from_splits(dat, id_tag)
                    use_split_column = True
                except Exception as e:
                    logger.warning(f"Failed to use predetermined splits: {e}. Falling back to random splits.")
                    use_split_column = False
            else:
                logger.info("No split column found. Using random splits.")
        
        # Fall back to random splits if predetermined splits not used or failed
        if not use_split_column:
            id_train, id_val, id_test = get_id_train_val_test(
                total_size=len(dat),
                split_seed=split_seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                n_train=n_train,
                n_test=n_test,
                n_val=n_val,
                keep_data_order=keep_data_order,
            )
        
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(output_dir, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]
    
    train_data, mean_train, std_train = get_pyg_dataset(
        dataset=dataset_train,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="train_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
    )
    val_data,_,_ = get_pyg_dataset(
        dataset=dataset_val,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="val_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
    )
    test_data,_,_ = get_pyg_dataset(
        dataset=dataset_test,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name=dataset,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="test_data",
        use_lattice=use_lattice,
        use_angle=use_angle,
        use_save=False,
        mean_train=mean_train,
        std_train=std_train,
    )

    
    collate_fn = train_data.collate
    if line_graph:
        collate_fn = train_data.collate_line_graph

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
        mean_train,
        std_train,
    )
    

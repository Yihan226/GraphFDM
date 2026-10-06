import numpy as np
import torch
from rdkit import Chem

from analysis.rdkit_functions import check_mol, correct_mol, mol2smiles


def _to_numpy_int(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def graph_cache_key(graph):
    node_types, bond_adj, pos_adj = graph
    node_types = _to_numpy_int(node_types)
    bond_adj = _to_numpy_int(bond_adj)
    pos_adj = _to_numpy_int(pos_adj)
    return (
        node_types.tobytes(),
        bond_adj.tobytes(),
        pos_adj.tobytes(),
        tuple(node_types.shape),
        tuple(bond_adj.shape),
        tuple(pos_adj.shape),
    )


def _prepare_graph_arrays(graph):
    node_types, bond_adj, pos_adj = graph
    node_types = _to_numpy_int(node_types)
    bond_adj = _to_numpy_int(bond_adj) - 1
    pos_adj = _to_numpy_int(pos_adj) - 1
    return node_types, bond_adj, pos_adj


def _canonicalize_largest_fragment(smiles):
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None
    if "." in smiles:
        parts = [part for part in smiles.split(".") if part]
        if len(parts) == 0:
            return None
        smiles = max(parts, key=len)
        mol = Chem.MolFromSmiles(smiles)
        return None if mol is None else Chem.MolToSmiles(mol)
    return smiles


def _check_raw_connected_validity(raw_mol):
    if raw_mol is None:
        return False
    try:
        raw_valid = check_mol(raw_mol, largest_connected_comp=False) is not None
        frags = Chem.rdmolops.GetMolFrags(raw_mol, sanitizeFrags=False)
        return bool(raw_valid and len(frags) == 1)
    except Exception:
        return False


def _decode_base_result(tokenizer, node_types, bond_adj, pos_adj):
    reward_smiles, _, _ = tokenizer.decode(
        node_types.tolist(),
        bond_adj.tolist(),
        pos_adj.tolist(),
    )
    reward_smiles = _canonicalize_largest_fragment(reward_smiles)
    try:
        raw_mol, _ = tokenizer.decode(
            node_types.tolist(),
            bond_adj.tolist(),
            pos_adj.tolist(),
            return_mol=True,
        )
    except Exception:
        raw_mol = None
    validity_nc = False
    validity_raw = False
    if raw_mol is not None:
        try:
            validity_nc = check_mol(raw_mol, largest_connected_comp=True) is not None
        except Exception:
            validity_nc = False
        validity_raw = _check_raw_connected_validity(raw_mol)
    return {
        "raw_mol": raw_mol,
        "reward_smiles": reward_smiles,
        "smiles": reward_smiles,
        "validity_nc": validity_nc,
        "validity_raw": validity_raw,
        "num_components": 0,
    }


def _apply_relaxed_validity(decoded_result, ensure_connected):
    result = dict(decoded_result)
    result["smiles"] = None
    result["num_components"] = 0
    mol = decoded_result.get("raw_mol")
    if mol is not None:
        try:
            working_mol = Chem.RWMol(mol)
            if ensure_connected:
                corrected_mol, _ = correct_mol(working_mol, connection=True)
            else:
                corrected_mol, _ = correct_mol(working_mol, connection=True)
                if corrected_mol is None:
                    corrected_mol, _ = correct_mol(working_mol, connection=False)
            smiles = mol2smiles(corrected_mol)
            corrected = Chem.MolFromSmiles(smiles) if smiles is not None else None
            if corrected is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(corrected, asMols=True, sanitizeFrags=True)
                result["num_components"] = len(mol_frags)
                largest_mol = max(mol_frags, default=corrected, key=lambda m: m.GetNumAtoms())
                largest_smiles = mol2smiles(largest_mol)
                if (
                    largest_mol is not None
                    and largest_smiles is not None
                    and len(largest_smiles) > 1
                    and Chem.MolFromSmiles(largest_smiles) is not None
                ):
                    result["smiles"] = largest_smiles
        except Exception:
            result["smiles"] = None

    return result


def decode_motif_graph(tokenizer, graph, cache=None):
    cache_key = None
    if cache is not None:
        cache_key = graph_cache_key(graph)
        if cache_key in cache:
            return cache[cache_key]

    node_types, bond_adj, pos_adj = _prepare_graph_arrays(graph)
    result = _decode_base_result(tokenizer, node_types, bond_adj, pos_adj)

    if cache is not None:
        cache[cache_key] = result
    return result


def decode_motif_graphs(tokenizer, graphs, cache=None):
    return [
        decode_motif_graph(
            tokenizer,
            graph,
            cache=cache,
        )
        for graph in graphs
    ]


def enrich_decoded_graph_with_validity(decoded_result, ensure_connected=True):
    return _apply_relaxed_validity(decoded_result, ensure_connected=ensure_connected)


def enrich_decoded_graphs_with_validity(decoded_results, ensure_connected=True):
    return [
        enrich_decoded_graph_with_validity(result, ensure_connected=ensure_connected)
        for result in decoded_results
    ]

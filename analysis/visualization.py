import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw


class MolecularVisualization:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos

    def _motif_label(self, motif_idx):
        try:
            motif_idx = int(motif_idx)
        except Exception:
            return str(motif_idx)
        vocab = getattr(self.dataset_infos, "atom_decoder", [])
        if 0 <= motif_idx < len(vocab):
            motif = str(vocab[motif_idx])
            if len(motif) > 12:
                motif = motif[:12] + "..."
            return f"{motif_idx}:{motif}"
        return str(motif_idx)

    def visualize_by_smiles(self, path, smiles_list, num_to_visualize):
        RDLogger.DisableLog("rdApp.*")
        os.makedirs(path, exist_ok=True)
        print(f"Visualizing corrected {num_to_visualize} of {len(smiles_list)}")
        if num_to_visualize > len(smiles_list):
            print(f"Shortening to {len(smiles_list)}")
            num_to_visualize = len(smiles_list)
        for i in range(num_to_visualize):
            smiles = smiles_list[i]
            if smiles is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            try:
                Draw.MolToFile(mol, os.path.join(path, f"molecule_corrected_{i}.png"))
            except Chem.rdchem.KekulizeException:
                print("Can't kekulize molecule")

    def visualize_chain_graph_step(self, output_path, node_types, bond_adj, pos_adj, title=None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        node_types = np.asarray(node_types, dtype=np.int64)
        bond_adj = np.asarray(bond_adj, dtype=np.int64)
        pos_adj = np.asarray(pos_adj, dtype=np.int64)

        graph = nx.Graph()
        for i, motif_idx in enumerate(node_types.tolist()):
            graph.add_node(i, label=self._motif_label(motif_idx))

        edge_labels = {}
        for i in range(len(node_types)):
            for j in range(i + 1, len(node_types)):
                bond_type = int(bond_adj[i, j])
                if bond_type <= 0:
                    continue
                graph.add_edge(i, j, weight=max(1, bond_type))
                pos_ij = int(pos_adj[i, j]) if i < pos_adj.shape[0] and j < pos_adj.shape[1] else 0
                pos_ji = int(pos_adj[j, i]) if j < pos_adj.shape[0] and i < pos_adj.shape[1] else 0
                edge_labels[(i, j)] = f"b{bond_type}|p{pos_ij}/{pos_ji}"

        plt.figure(figsize=(6, 6))
        if graph.number_of_nodes() == 0:
            plt.text(0.5, 0.5, "Empty graph", ha="center", va="center")
            plt.axis("off")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            return

        pos_layout = nx.spring_layout(graph, seed=0)
        nx.draw_networkx_nodes(graph, pos_layout, node_color="#d9e8fb", edgecolors="#274690", node_size=1200)
        nx.draw_networkx_labels(
            graph,
            pos_layout,
            labels={node: data["label"] for node, data in graph.nodes(data=True)},
            font_size=7,
        )
        if graph.number_of_edges() > 0:
            widths = [graph[u][v]["weight"] for u, v in graph.edges()]
            nx.draw_networkx_edges(graph, pos_layout, width=widths, edge_color="#576490")
            nx.draw_networkx_edge_labels(graph, pos_layout, edge_labels=edge_labels, font_size=6)

        plt.title(title or "Graph Chain Step")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()

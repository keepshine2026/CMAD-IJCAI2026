import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio

from CMAD.graph.node import Node
from CMAD.agents.agent_registry import AgentRegistry
from CMAD.prompt.prompt_set_registry import PromptSetRegistry
from CMAD.llm.profile_embedding import get_sentence_embedding
from CMAD.gnn.gcn import GCN, MLP
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        self.gcn = GCN(self.features.size(1)*2,16,self.features.size(1))
        self.mlp = MLP(384,16,16)

        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        # self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
        #                                          requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
    
    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes,num_nodes))
        role_2_id = {}
        
        for edge in role_connect:
            in_role, out_role = edge
            role_2_id[in_role] = []
            role_2_id[out_role] = []
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)
            
        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1
        
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
    
    def construct_new_features(self, query):
        # Keep all tensors on the same device to support GPU training/eval.
        try:
            device = next(self.gcn.parameters()).device
        except Exception:
            device = self.features.device
        if isinstance(self.features, torch.Tensor) and self.features.device != device:
            self.features = self.features.to(device=device)
        if isinstance(self.role_adj_matrix, torch.Tensor) and self.role_adj_matrix.device != device:
            self.role_adj_matrix = self.role_adj_matrix.to(device=device)

        query_embedding = torch.tensor(get_sentence_embedding(query), dtype=self.features.dtype, device=device)
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes, 1))
        new_features = torch.cat((self.features, query_embedding), dim=1)
        return new_features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        device = self.spatial_logits.device if isinstance(getattr(self, "spatial_logits", None), torch.Tensor) else torch.device("cpu")
        dtype = self.spatial_logits.dtype if isinstance(getattr(self, "spatial_logits", None), torch.Tensor) else torch.float32
        log_probs = [torch.zeros((), device=device, dtype=dtype, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold is not None:
                    edge_prob = (edge_prob > float(threshold)).to(dtype=edge_prob.dtype)
                if torch.rand_like(edge_prob) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))

    def construct_spatial_connection_all(self, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_spatial_connection()
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                out_node.add_successor(in_node,'spatial')
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        device = self.temporal_logits.device if isinstance(getattr(self, "temporal_logits", None), torch.Tensor) else torch.device("cpu")
        dtype = self.temporal_logits.dtype if isinstance(getattr(self, "temporal_logits", None), torch.Tensor) else torch.float32
        log_probs = [torch.zeros((), device=device, dtype=dtype, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold is not None:
                edge_prob = (edge_prob > float(threshold)).to(dtype=edge_prob.dtype)
            if torch.rand_like(edge_prob) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))

    def to_pyg_graph(self, input: Dict[str, Any], keep_all_edge=True) -> Data:
        # Export graph structure into a torch_geometric.data.Data object.
        agent_nodes = [node for node in self.nodes.values() if node != self.decision_node]
        node_id_to_idx = {node.id: idx for idx, node in enumerate(agent_nodes)}

        node_features = []
        for node in agent_nodes:
            role = node.role
            constraint = node.constraint if hasattr(node, 'constraint') else "No constraint"
            feature = {'role': role, 'constraint': constraint}
            node_features.append(feature)

        if keep_all_edge:
            self.spatial_logits = torch.ones(len(agent_nodes) ** 2)
            self.construct_spatial_connection_all()
        else:
            new_features = self.construct_new_features(input['task'])
            logits = self.gcn(new_features, self.role_adj_matrix)
            logits = self.mlp(logits)
            self.spatial_logits = logits @ logits.t()
            self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))
            self.construct_spatial_connection()

        edge_indices = []
        edge_weights = []
        for src_node in agent_nodes:
            src_idx = node_id_to_idx[src_node.id]
            for dst_node in src_node.spatial_successors:
                if dst_node.id in node_id_to_idx:
                    dst_idx = node_id_to_idx[dst_node.id]
                    edge_indices.append([src_idx, dst_idx])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=input['task']
        )

    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        new_features = self.construct_new_features(input['task'])
        logits = self.gcn(new_features,self.role_adj_matrix)
        logits = self.mlp(logits)
        self.spatial_logits = logits @ logits.t()
        self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks

    def get_run_metadata(self) -> Dict[str, Any]:
        """
        Lightweight run metadata used by experiment scripts to log transcripts/edges/nodes.

        NOTE: This version does not rely on any internal logging during execution; it reconstructs
        edge logs from the *current* sampled topology and each source node's latest output.
        This is sufficient for num_rounds=1 (the common case in this repo).
        """

        def _last_output_text(node: Node) -> str:
            outs = getattr(node, "outputs", None)
            if isinstance(outs, list):
                if not outs:
                    return ""
                last = outs[-1]
            else:
                last = outs
            return "" if last is None else str(last)

        edge_logs: List[Dict[str, Any]] = []
        for src_id, src_node in self.nodes.items():
            src_role = getattr(src_node, "role", "Unknown")
            msg_text = _last_output_text(src_node)
            for dst_node in getattr(src_node, "spatial_successors", []) or []:
                dst_id = str(getattr(dst_node, "id", "") or "")
                if not dst_id or dst_id not in self.nodes:
                    continue
                dst_role = getattr(dst_node, "role", "Unknown")
                edge_logs.append(
                    {
                        "round": 0,
                        "edge_type": "spatial",
                        "src_id": str(src_id),
                        "src_role": str(src_role),
                        "dst_id": str(dst_id),
                        "dst_role": str(dst_role),
                        "timestamp": None,
                        "message": msg_text,
                    }
                )
            for dst_node in getattr(src_node, "temporal_successors", []) or []:
                dst_id = str(getattr(dst_node, "id", "") or "")
                if not dst_id or dst_id not in self.nodes:
                    continue
                dst_role = getattr(dst_node, "role", "Unknown")
                edge_logs.append(
                    {
                        "round": 0,
                        "edge_type": "temporal",
                        "src_id": str(src_id),
                        "src_role": str(src_role),
                        "dst_id": str(dst_id),
                        "dst_role": str(dst_role),
                        "timestamp": None,
                        "message": msg_text,
                    }
                )

        node_topology: List[Dict[str, Any]] = []
        for node_id, node in self.nodes.items():
            preds = [p for p in getattr(node, "spatial_predecessors", []) or [] if getattr(p, "id", None) in self.nodes]
            succs = [s for s in getattr(node, "spatial_successors", []) or [] if getattr(s, "id", None) in self.nodes]
            node_topology.append(
                {
                    "round": 0,
                    "node_id": str(node_id),
                    "role": str(getattr(node, "role", "Unknown")),
                    "in_degree": int(len(preds)),
                    "out_degree": int(len(succs)),
                    "predecessor_roles": [str(getattr(p, "role", "Unknown")) for p in preds],
                    "successor_roles": [str(getattr(s, "role", "Unknown")) for s in succs],
                    "laplacian_embedding": None,
                }
            )

        return {
            "edge_logs": edge_logs,
            # different scripts use different keys; keep aliases for backwards compatibility
            "node_topology": node_topology,
            "node_logs": node_topology,
            "node_logs_topology": node_topology,
        }


class TestGraph(ABC):
    def __init__(
            self,
            domain: str,
            llm_name: Optional[str],
            decision_method: str,
            pyg_data: Data,
            node_kwargs: List[Dict] = None,
    ):
        self.domain = domain
        self.llm_name = llm_name
        self.decision_node = AgentRegistry.get(
            decision_method, **{"domain": domain, "llm_name": llm_name}
        )
        self.nodes = {}
        self._build_from_pyg(pyg_data)

    def _build_from_pyg(self, pyg_data: Data):

        roles = [d['role'] for d in pyg_data.x]
        constraints = [d.get('constraint') for d in pyg_data.x]
        if self.domain == 'mmlu':
            agent_type = 'AnalyzeAgent'
        elif self.domain == 'humaneval':
            agent_type = 'CodeWriting'
        elif self.domain == 'gsm8k':
            agent_type = 'MathSolver'
        elif self.domain == 'aqua':
            agent_type = 'MathSolver'
        prompt_set = PromptSetRegistry.get(self.domain)

        for idx, (role, constraint) in enumerate(zip(roles, constraints)):
            actual_constraint = prompt_set.get_description(role)

            agent_kwargs = {
                'role': role,
                'domain': self.domain,
                'llm_name': self.llm_name
            }
            if self.domain == 'mmlu':
                agent_kwargs['constraint'] = actual_constraint
            agent = AgentRegistry.get(
                agent_type,
                **agent_kwargs
            )
            self.add_node(agent)

        edge_index = pyg_data.edge_index.cpu().numpy().T
        node_list = list(self.nodes.values())

        for src_idx, dst_idx in edge_index:
            src_node = node_list[src_idx]
            dst_node = node_list[dst_idx]
            src_node.add_successor(dst_node, "spatial")

    def add_node(self, node: Node):
        node_id = node.id if node.id else shortuuid.ShortUUID().random(length=4)
        self.nodes[node_id] = node
        return node

    def set_blocked_spatial_edges_by_indices(self, index_pairs: List[Tuple[int, int]]):
        """
        Remove spatial edges specified by (src_idx, dst_idx) in the pyg node order.
        Used by counterfactual generators.
        """
        node_list = list(self.nodes.values())
        for src_idx, dst_idx in index_pairs or []:
            if src_idx is None or dst_idx is None:
                continue
            if int(src_idx) < 0 or int(dst_idx) < 0:
                continue
            if int(src_idx) >= len(node_list) or int(dst_idx) >= len(node_list):
                continue
            src_node = node_list[int(src_idx)]
            dst_node = node_list[int(dst_idx)]
            if dst_node in getattr(src_node, "spatial_successors", []):
                src_node.remove_successor(dst_node, "spatial")

    def get_run_metadata(self) -> Dict[str, Any]:
        def _last_output_text(node: Node) -> str:
            outs = getattr(node, "outputs", None)
            if isinstance(outs, list):
                if not outs:
                    return ""
                last = outs[-1]
            else:
                last = outs
            return "" if last is None else str(last)

        node_by_id: Dict[str, Node] = self.nodes or {}
        edge_logs: List[Dict[str, Any]] = []
        for src_id, src_node in node_by_id.items():
            src_role = getattr(src_node, "role", "Unknown")
            msg_text = _last_output_text(src_node)
            for dst_node in getattr(src_node, "spatial_successors", []) or []:
                dst_id = str(getattr(dst_node, "id", "") or "")
                if not dst_id or dst_id not in node_by_id:
                    continue
                dst_role = getattr(dst_node, "role", "Unknown")
                edge_logs.append(
                    {
                        "round": 0,
                        "edge_type": "spatial",
                        "src_id": str(src_id),
                        "src_role": str(src_role),
                        "dst_id": str(dst_id),
                        "dst_role": str(dst_role),
                        "timestamp": None,
                        "message": msg_text,
                    }
                )
            for dst_node in getattr(src_node, "temporal_successors", []) or []:
                dst_id = str(getattr(dst_node, "id", "") or "")
                if not dst_id or dst_id not in node_by_id:
                    continue
                dst_role = getattr(dst_node, "role", "Unknown")
                edge_logs.append(
                    {
                        "round": 0,
                        "edge_type": "temporal",
                        "src_id": str(src_id),
                        "src_role": str(src_role),
                        "dst_id": str(dst_id),
                        "dst_role": str(dst_role),
                        "timestamp": None,
                        "message": msg_text,
                    }
                )

        node_topology: List[Dict[str, Any]] = []
        for node_id, node in node_by_id.items():
            preds = [p for p in getattr(node, "spatial_predecessors", []) or [] if getattr(p, "id", None) in node_by_id]
            succs = [s for s in getattr(node, "spatial_successors", []) or [] if getattr(s, "id", None) in node_by_id]
            node_topology.append(
                {
                    "round": 0,
                    "node_id": str(node_id),
                    "role": str(getattr(node, "role", "Unknown")),
                    "in_degree": int(len(preds)),
                    "out_degree": int(len(succs)),
                    "predecessor_roles": [str(getattr(p, "role", "Unknown")) for p in preds],
                    "successor_roles": [str(getattr(s, "role", "Unknown")) for s in succs],
                    "laplacian_embedding": None,
                }
            )

        return {
            "edge_logs": edge_logs,
            "node_topology": node_topology,
            "node_logs": node_topology,
            "node_logs_topology": node_topology,
        }

    async def arun(self, inputs: Dict[str, Any], num_rounds=1, max_tries: int = 3, max_time: int = 600) -> List[Any]:

        for round in range(num_rounds):

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if
                                    deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(inputs),
                                               timeout=max_time)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        self.connect_decision_node()
        await self.decision_node.async_execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers

    def connect_decision_node(self):
        for node in self.nodes.values():
            node.add_successor(self.decision_node)

    def update_memory(self):
        for id, node in self.nodes.items():
            node.update_memory()

    def construct_temporal_connection(self, round: int = 0, temperature: float = 1.0,
                                      threshold: float = None, ):
        self.clear_temporal_connection()

        if round == 0:
            return 0

    def clear_temporal_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    

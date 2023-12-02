from __future__ import annotations

import functools
import logging

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torchmetrics
from envs import shipping_assignment_state
from envs.shipping_assignment_env import ShippingAssignmentEnvironment
from torch import Tensor
from torch_geometric.nn import GCNConv, max_pool, global_max_pool

from agents.Agent import Agent

# Taken from a legacy implementation in the environment, to avoid the import. It's the location on the "state_vector" of the customer ID.
from agents.optimizer_agents import LookaheadAgent
from dqn.noisy_linear import NoisyLinear

_CUSTOMER_METADATA_NEURON = -3


class PyTorchAgent(Agent):
    """
    Base PyTorch Agent class for agents in Seminario II (~Sep 13).
    It's expected that an agent with this impl passes a network module that generates
    Q values the size of the environment action space.

    The agent is also responsible for converting the state into an input tensor. The
    default is "state_vector" in the state named tuple.

    Should override: get_state_vector, get_action, train.

    Args:
        env: training environment
        net: The PyTorch network module.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        env: ShippingAssignmentEnvironment,
        net,
        epsilon,
        device="cpu",
    ) -> None:
        super().__init__(env)
        self.env = env
        self.net = net
        self.device = device
        self.epsilon = epsilon
        self.dcs_per_customer_array = self.env.physical_network.dcs_per_customer_array
        self.invalid_penalty = self.env.physical_network.big_m_cost
        # As of Nov 20, this is a critical thing to set for neural net agents, manually being set in the PTL init.
        self.ref_tensor = None  # WARNING: if this is not set sometime before calling forward, code will fail!!!!

    def get_state_vector(self, state):
        """Must be implemented by the concrete agent"""
        pass

    def mask_q_values(self, q_values, state):
        # Todo maybe move to the action space.
        customer_node_id = state.open[0].customer.node_id
        customer_id = self.env.physical_network.get_customer_id(customer_node_id)
        customer_valid_dcs = self.dcs_per_customer_array[customer_id, :]
        # Penalty logic: 1-valid dcs gives you invalid. Multiply that to amplify the 1s to penalty.
        penalty_for_invalid = -((1 - customer_valid_dcs) * self.invalid_penalty)
        masked_q_values = q_values + torch.tensor(
            penalty_for_invalid, device=q_values.device
        )
        return masked_q_values

    def get_action(self, state) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy


        Returns:
            action
        """
        # TODO Sep 13: Before copying, this code was called in a method of this class called play step, decorated with no_grad.
        # Should I still no grad my stuff when this is called? Possibly, when doing the env step in the runner.
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample(state.open[0].customer.node_id)
        else:
            state_vector = self.get_state_vector(state)
            if isinstance(state_vector, np.ndarray):
                state_vector = torch.tensor([state_vector]).type_as(self.ref_tensor)
            elif isinstance(state_vector, torch.Tensor):
                state_vector = state_vector.type_as(self.ref_tensor)

            # Getting the action with the highest Q value.
            q_values = self.net(
                state_vector
            )  # TODO check that the star OP works with regular NN

            masked_q_values = self.mask_q_values(q_values, state)

            self.logger.debug("Network output Q values")
            self.logger.debug(q_values)
            self.logger.debug("Masked")
            self.logger.debug(masked_q_values)
            _, action = torch.max(masked_q_values, dim=1)
            action = int(action.item())

        self.logger.debug(f"Agent chose action {action}")

        return action

    def reset(self) -> None:
        """TODO  as of sep 13 idk if I need this anymore (was copy pasted from old ptl agent), but probably was from OpenAI impl"""
        pass

    def train(self, experience):
        pass


class CustomerOnehotDQN(nn.Module):
    """
    Simple MLP network that uses the one hot encoding of customer IDs.

    Args:
        num_customers: observation size, which is the total number of customers
        num_dcs: action space, which is the total number of DCs.
    """

    def __init__(self, num_customers: int, num_dcs: int):
        super(CustomerOnehotDQN, self).__init__()
        self.num_customers = num_customers
        self.num_dcs = num_dcs
        # Shallow.
        # self.net = nn.Sequential(
        #     nn.Linear(self.num_customers, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, self.num_dcs),
        # )

        # ultra deep
        self.net = nn.Sequential(
            nn.Linear(self.num_customers, 128),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(8, self.num_dcs),
        )

        # Check shipping_assignment_environment for the metadata neuron definition (create_state_vector, as of Sep 18)
        self.customer_metadata_neuron = _CUSTOMER_METADATA_NEURON

    def forward(self, x):
        """Convert the traditional state vector into one hot encoding of the customer."""
        with torch.no_grad():
            xp = torchmetrics.utilities.data.to_onehot(
                x[:, self.customer_metadata_neuron] - self.num_dcs,
                num_classes=self.num_customers,
            )

        return self.net(xp.float())


class CustomerDQNAgent(PyTorchAgent):
    def __init__(self, env, customer_dqn, epsilon, **kwargs):
        super().__init__(env, customer_dqn, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        return state.state_vector.reshape(-1)

    def train(self, experience):
        """This is not needed, it's handled by PTL"""
        pass


class MaskedMLPDQN(nn.Module):
    """An MLP That takes as an input a one hot encoding mask of the valid warehouses.
    The motivation is that the agent doesn't have to learn tnat information and can
     instead focus on which is the best warehouse in terms of optimization cost.
    """

    def __init__(self, num_dcs):
        super().__init__()
        self.num_dcs = num_dcs
        self.net = nn.Sequential(
            nn.Linear(self.num_dcs, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_dcs),
        )

    def forward(self, x):
        return self.net(x.float())


class MaskedMLPDQNAgent(PyTorchAgent):
    """An MLP whose input is the mask of which DCS are valid."""

    def __init__(self, env, mlp_dqn, epsilon, **kwargs):
        super().__init__(env, mlp_dqn, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        latest_open_order = state.open[0]
        customer_id = state.physical_network.get_customer_id(
            latest_open_order.customer.node_id
        )
        return state.physical_network.dcs_per_customer_array[customer_id, :]

    def train(self, experience):
        """This is not needed, it's handled by PTL"""
        pass


class DebugMaskedMLPCheatDQN(nn.Module):
    """Debug MLP with cheat input from looakehead actions."""

    def __init__(self, num_dcs):
        super().__init__()
        self.num_dcs = num_dcs
        # The input of  this cheat network is two onehots of |DC|
        # self.net = nn.Sequential(
        #     nn.Linear(self.num_dcs, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, self.num_dcs),
        # )
        # self.net = nn.Sequential(
        #     nn.Linear(self.num_dcs, self.num_dcs * 64),
        #     #nn.ReLU(),
        #     nn.Tanh(),
        #     nn.Linear(self.num_dcs * 64, self.num_dcs),
        # )

        self.net = nn.Sequential(
            nn.Linear(self.num_dcs, self.num_dcs),
        )

    def forward(self, x):
        return self.net(x.float())


class MaskedMLPWithCheatDQNAgent(PyTorchAgent):
    """This agent is to debug that the NNs are actually learning, because the lookahead input
    is a cheat code and it should use it to get the best action most times."""

    logger = logging.getLogger(__name__)

    def __init__(self, env, mlp_dqn, epsilon, **kwargs):
        self.lookahead_agent = LookaheadAgent(env)
        super().__init__(env, mlp_dqn, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        lookahead_action = self.lookahead_agent.get_action(
            state
        )  # Todo: this calls too many lookaheads.
        # TODO also consider using the cost vector directly.
        lookahead_onehot = np.zeros(state.physical_network.num_dcs) * 0.0
        lookahead_onehot[lookahead_action] = 1.0
        latest_open_order = state.open[0]
        customer_id = state.physical_network.get_customer_id(
            latest_open_order.customer.node_id
        )
        mask_vector = state.physical_network.dcs_per_customer_array[customer_id, :]

        # state_vector = np.hstack((mask_vector, lookahead_onehot))
        state_vector = lookahead_onehot

        self.logger.debug("MLP with Cheat state vector (lookahead onehotted)")
        self.logger.debug(lookahead_onehot)
        self.logger.debug(" and valid warehouses are: ")
        self.logger.debug(mask_vector)

        return lookahead_onehot

    def train(self, experience):
        """This is not needed, it's handled by PTL"""
        pass


class MaskedPlusOneHotDQN(nn.Module):
    """The combination of MaskedNLP and CustomerOneHot"""

    def __init__(self, num_customers, num_dcs):
        super().__init__()
        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.net = nn.Sequential(
            nn.Linear(self.num_dcs + self.num_customers, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_dcs),
        )

    def forward(self, x):
        return self.net(x.float())


class MaskedPlusOneHotDQNAgent(PyTorchAgent):
    """An MLP whose input is the mask of which DCS are valid, concatenated with customer onehot."""

    def __init__(self, env, mlp_dqn, epsilon, **kwargs):
        super().__init__(env, mlp_dqn, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        return (
            state_to_mask_concat_onehot(state).detach().cpu().numpy()
        )  # TODO inefficient as hell.

    def train(self, experience):
        pass


class FullMLPDQN(nn.Module):
    """The combination of MaskedNLP and CustomerOneHot"""

    def __init__(self, num_customers, num_dcs, num_commodities):
        super().__init__()
        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.num_commodities = num_commodities
        # Calculating input size.
        mask_size = num_dcs
        onehot_size = num_customers
        inventory_size = num_commodities * num_dcs
        demand = num_commodities
        backlog = num_commodities
        inventory_after_current_order = num_dcs  # A vector of size |W| that says if the current order fits in each dc.
        input_size = (
            mask_size
            + onehot_size
            + inventory_size
            + demand
            + backlog
            + inventory_after_current_order
        )
        # Don't be fooled, the layer sizes were pretty arbitrary.
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, input_size * 4),
        #     nn.LayerNorm(
        #         input_size * 4
        #     ),  # TODO dont understand why batchnorm1d dont work, probably some shape think. Look for the diff between these two.
        #     nn.ReLU(),
        #     nn.Linear(input_size * 4, input_size * 2),
        #     nn.LayerNorm(
        #         input_size * 2
        #     ),  # TODO dont understand why batchnorm1d dont work, probably some shape think. Look for the diff between these two.
        #     nn.ReLU(),
        #     nn.Linear(input_size * 2, input_size),
        #     nn.LayerNorm(
        #         input_size
        #     ),  # TODO dont understand why batchnorm1d dont work, probably some shape think. Look for the diff between these two.
        #     nn.ReLU(),
        #     nn.Linear(input_size, input_size // 2),
        #     nn.LayerNorm(
        #         input_size // 2
        #     ),  # TODO dont understand why batchnorm1d dont work, probably some shape think. Look for the diff between these two.
        #     nn.ReLU(),
        #     nn.Linear(input_size // 2, self.num_dcs),
        # )
        # Small wide
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, 256), nn.Tanh(), nn.Linear(256, self.num_dcs)
        # )
        # Linear
        # self.net = nn.Sequential(nn.Linear(input_size, self.num_dcs))
        # Small wide noisy
        # self.net = nn.Sequential(
        #     NoisyLinear(input_size, 256), nn.Tanh(), NoisyLinear(256, self.num_dcs)
        # )
        # Linear Noisy
        self.net = nn.Sequential(nn.Linear(input_size, self.num_dcs))

    def forward(self, x):
        normalized_in = nn.functional.normalize(x.float())

        return self.net(normalized_in)


class FullMLPDQNAgent(PyTorchAgent):
    def __init__(self, env, mlp_dqn, epsilon, **kwargs):
        super().__init__(env, mlp_dqn, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        mask_and_onehot = state_to_mask_concat_onehot(state)
        inventory_vector = state_to_inventory_vector(state)

        latest_open_order = state.open[0]
        order_demand_vector = torch.tensor(latest_open_order.demand)

        # Get size of commodities form the only order we're guaranteed exists.
        num_commodities = latest_open_order.demand.shape[0]

        # fixed_demand = orders_to_demand_summary(state.fixed, num_commodities)
        fixed_demand = shipping_assignment_state.state_to_fixed_demand(state)
        fixed_demand_per_dc = (
            shipping_assignment_state.state_to_demand_per_warehouse_commodity(state)
        )
        open_demand = orders_to_demand_summary(state.open[1:], num_commodities)

        aggregate_demand_vector = torch.tensor(fixed_demand + open_demand)
        inventory_minus_open_order = (
            inventory_vector - fixed_demand_per_dc - order_demand_vector
        )
        with torch.no_grad():
            full_vector = (
                torch.cat(
                    [
                        mask_and_onehot,
                        inventory_vector,
                        order_demand_vector,
                        aggregate_demand_vector,
                        inventory_minus_open_order,
                    ]
                )
                .detach()
                .cpu()
                .numpy()
            )
            return full_vector  # no type as because it's done down the road for state_vectors of type ndarray.

    def train(self, experience):
        pass


class MaskPlusConsumptionMLP(nn.Module):
    """The combination of MaskedNLP and the consumption (inventory after fixed orders and current)"""

    def __init__(self, num_customers, num_dcs, num_commodities):
        super().__init__()
        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.num_commodities = num_commodities
        # Calculating input size.
        mask_size = num_dcs
        inventory_after_current_order = (
            num_dcs * num_commodities
        )  # A vector of size |W| that says if the current order fits in each dc.
        input_size = mask_size + inventory_after_current_order
        # Small wide
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, 256), nn.Tanh(), nn.Linear(256, self.num_dcs)
        # )
        # Linear
        # self.net = nn.Sequential(nn.Linear(input_size, self.num_dcs))
        # Small wide noisy
        # self.net = nn.Sequential(
        #     NoisyLinear(input_size, 256), nn.Tanh(), NoisyLinear(256, self.num_dcs)
        # )
        # Medium wide with 3 noisies (not as good as 2)
        # self.net = nn.Sequential(
        #     NoisyLinear(input_size, 512),
        #     nn.Tanh(),
        #     NoisyLinear(512, 256),
        #     nn.Tanh(),
        #     NoisyLinear(256, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, self.num_dcs),
        # )
        # Wide with 2 noisies, better than just 1 and also as good as 3.
        self.net = nn.Sequential(
            NoisyLinear(input_size, 512),
            nn.Tanh(),
            NoisyLinear(512, 256),
            nn.Tanh(),
            nn.Linear(256, self.num_dcs),
        )
        # Linear Noisy
        # self.net = nn.Sequential(nn.Linear(input_size, self.num_dcs))

    def forward(self, x):
        normalized_in = nn.functional.normalize(x.float())

        return self.net(normalized_in)


class MaskPlusConsumptionMLPAgent(PyTorchAgent):
    def __init__(self, env, mlp_dqn, epsilon, **kwargs):
        super().__init__(env, mlp_dqn, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        mask = torch.tensor(state_to_mask(state)).type_as(self.ref_tensor)
        inventory_vector = state_to_inventory_vector(state)

        fixed_demand_per_dc = torch.tensor(
            shipping_assignment_state.state_to_demand_per_warehouse_commodity(state)
        ).type_as(self.ref_tensor)

        latest_open_order = state.open[0]
        order_demand_vector = torch.tensor(latest_open_order.demand).type_as(
            self.ref_tensor
        )

        num_dcs = state.physical_network.num_dcs
        num_commodities = state.physical_network.num_commodities
        # This line was pretty much trial and error tensor ops:
        # Expand requires a matrix that the "singleton dim" (dim of 1) matches in size, and if I do
        # .reshape(-1,1) later I can't stack it as [a,b,c,a,b,c]. That's why the (1,-1 reshape).
        # The result is a vector of size |W|*|K| which is the order demand repeated |W| times.
        rep_order_demand = (
            order_demand_vector.reshape(1, -1)
            .expand(num_dcs, num_commodities)
            .reshape(-1)
        )

        # the available feature is inventory - fixed at each DC - the current order's demand subtracted to all warehouses.
        inventory_after_order = (
            torch.tensor(inventory_vector).type_as(self.ref_tensor)
            - fixed_demand_per_dc
            - rep_order_demand
        )
        with torch.no_grad():
            mask = nn.functional.normalize(mask.float().reshape(1, -1)).flatten()
            inventory_after_order = nn.functional.normalize(
                inventory_after_order.float().reshape(1, -1)
            ).flatten()
            full_vector = (
                torch.cat(
                    [
                        mask,
                        inventory_after_order,
                    ]
                )
                .detach()
                .cpu()
                .numpy()
            )

            return full_vector

    def train(self, experience):
        pass


class PhysnetAggDemandGCN(torch.nn.Module):
    """
    A GCN where every feature vector is a vector of commodity demands.
    For inventories, it's a positive vector of available units.
    For demand, it's negative sum units of all orders in the horizon for that customer.

    """

    def __init__(self, num_commodities, num_dcs, num_customers, hidden_units):
        """Hidden units: min recommended its 16, and more,relative to num dcs."""
        super().__init__()
        self.conv1 = GCNConv(num_commodities, hidden_units)
        self.conv2 = GCNConv(hidden_units, hidden_units // 2)
        # GraphPool to get (batch,num_commodities)
        self.mlp = nn.Linear(hidden_units // 2, num_dcs)
        self.normalize_inputs = False

    def forward(self, data: torch_geometric.data.Data) -> Tensor:
        """
        Args: #Todo update docs
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity matrix of shape [2, num_edges]
        Returns: [batch, num_dcs]
        """
        dx, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )  # If not using batches, all nodes in Data should map to the same batch.
        if (
            batch is None
        ):  # We're not doing batch inference so all nodes belong to the same graph
            # TODO important: if I were to use torch.ones instead of zeros, pyg assumes there are two graphs and will leave one empty
            batch = torch.zeros(dx.shape[0]).long()
        if self.normalize_inputs:
            x = nn.functional.normalize(dx.float().reshape(1, -1)).reshape(-1, 1)
        else:
            x = dx.float()
        bx = self.conv1(x, edge_index).relu()
        cx = self.conv2(bx, edge_index).relu()
        # Todo only tested in non batch. But seems legit, stacks all features vertically for each node.
        px = global_max_pool(cx, batch)
        mx = self.mlp(px)
        return mx


class PhysnetAggDemandGCNAgent(PyTorchAgent):
    def __init__(self, env, gcn_module, epsilon, **kwargs):
        super().__init__(env, gcn_module, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        # Node features are agg balance in horizon for each physical node
        graph_data = shipping_assignment_state.state_to_agg_balance_in_horizon_gnn(
            state
        )
        graph_data.x = graph_data.x.type_as(self.ref_tensor)
        graph_data.edge_index = graph_data.edge_index.type_as(self.ref_tensor).long()
        # We also need the adjacency matrix.

        return graph_data

    def train(self, experience):
        pass


class AggDemandAndNodeTypeGraphlevelGCN(torch.nn.Module):
    """
    A GCN where every feature vector is a vector of commodity demands.
    For inventories, it's a positive vector of available units.
    For demand, it's negative sum units of all orders in the horizon for that customer.

    """

    def __init__(self, num_commodities, num_dcs, num_customers, hidden_units):
        """Hidden units: min recommended its 16, and more,relative to num dcs."""
        super().__init__()
        self.conv1 = GCNConv(num_commodities + 2, hidden_units)
        self.conv2 = GCNConv(hidden_units, hidden_units // 2)
        # GraphPool to get (batch,num_commodities)
        self.mlp = nn.Linear(hidden_units // 2, num_dcs)
        self.normalize_inputs = False

    # def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
    def forward(self, data: torch_geometric.data.Data) -> Tensor:
        """
        Args: #Todo update docs
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity matrix of shape [2, num_edges]
        Returns: [batch, num_dcs]
        """
        dx, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )  # If not using batches, all nodes in Data should map to the same batch.

        if (
            batch is None
        ):  # We're not doing batch inference so all nodes belong to the same graph
            # TODO important: if I were to use torch.ones instead of zeros, pyg assumes there are two graphs and will leave one empty
            batch = torch.zeros(dx.shape[0]).type_as(dx).long()
        else:
            data.batch = data.batch.type_as(dx).long()
        if self.normalize_inputs:
            x = nn.functional.normalize(dx.float().reshape(1, -1)).reshape(-1, 1)
        else:
            x = dx.float()
        bx = self.conv1(x, edge_index).relu()
        cx = self.conv2(bx, edge_index).relu()
        # Todo only tested in non batch. But seems legit, stacks all features vertically for each node.
        px = global_max_pool(cx, batch)
        mx = self.mlp(px)
        return mx


class AggDemandAndNodeTypeGraphlevelGCNAgent(PyTorchAgent):
    def __init__(self, env, gcn_module, epsilon, **kwargs):
        super().__init__(env, gcn_module, epsilon=epsilon, **kwargs)

    def get_state_vector(self, state):
        # Node features are agg balance in horizon for each physical node
        graph_data = shipping_assignment_state.state_to_agg_balance_in_horizon_gnn(
            state
        )
        node_type_data = shipping_assignment_state.state_to_node_type_graphmarkers(
            state
        )
        # TODO comp is slow didnt test anything cant work like this, check that cat is correct
        with torch.no_grad():
            graph_data.x = torch.hstack(
                [graph_data.x, torch.tensor(node_type_data)]
            ).type_as(self.ref_tensor)
            graph_data.edge_index = graph_data.edge_index.type_as(
                self.ref_tensor
            ).long()
        # We also need the adjacency matrix.

        return graph_data

    def train(self, experience):
        pass


# Utility funcs used by many agents.
def customer_id_to_onehot(customer_id, num_customers) -> torch.Tensor:
    # TODO might be more efficient to just use numpy. Also document shape
    with torch.no_grad():
        return torchmetrics.utilities.data.to_onehot(
            torch.tensor([customer_id]),
            num_classes=num_customers,
        )


def state_to_mask(state):
    """Gets the mask of valid DCs for the latest order"""
    latest_open_order = state.open[0]
    customer_id = state.physical_network.get_customer_id(
        latest_open_order.customer.node_id
    )
    mask = state.physical_network.dcs_per_customer_array[customer_id, :]
    return mask


def state_to_mask_concat_onehot(state) -> torch.Tensor:
    """Converts a state to a valid warehouse mask concat with customer onehot"""
    latest_open_order = state.open[0]
    customer_id = state.physical_network.get_customer_id(
        latest_open_order.customer.node_id
    )
    num_customers = state.physical_network.num_customers

    mask = state.physical_network.dcs_per_customer_array[customer_id, :]
    onehot_vector = customer_id_to_onehot(customer_id, num_customers)
    with torch.no_grad():
        return torch.cat([onehot_vector.reshape(-1), torch.tensor(mask)])


def state_to_inventory_vector(state):
    return torch.tensor(state.inventory.reshape(-1)).detach()


def orders_to_demand_summary(orders, num_commodities):
    """Summarizes order demand"""
    demand_vectors = [o.demand for o in orders]
    return functools.reduce(
        lambda a, b: a + b, demand_vectors, np.zeros(num_commodities)
    )

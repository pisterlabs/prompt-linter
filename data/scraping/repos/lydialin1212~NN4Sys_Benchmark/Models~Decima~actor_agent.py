import numpy as np
import torch
import torch.nn as nn

import bisect
from param import *
from utils import *
from tf_op import *
from msg_passing_path import *
from gcn import GraphCNN
from gsn import GraphSNN
from agent import Agent
from spark_env.job_dag import JobDAG
from spark_env.node import Node
import time

PATH = "./result/"

cuda = torch.cuda.is_available()
cuda = False
print("cuda", cuda)
device = torch.device("mps")
mps = False


class ActorNetwork(nn.Module):
    def __init__(self, node_input_dim, job_input_dim, output_dim, executor_levels):
        # torch layers
        super().__init__()

        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.output_dim = output_dim
        self.executor_levels = executor_levels

        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.fc5 = nn.Linear(20, 32)
        self.act_fn = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, node_inputs, gcn_outputs, job_inputs,
                gsn_dag_summary, gsn_global_summary,
                node_valid_mask, job_valid_mask,
                gsn_summ_backward_map):
        # takes output from graph embedding and raw_input from environment

        batch_size = node_valid_mask.size()[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = node_inputs.view([batch_size, -1, self.node_input_dim])

        # (2) reshape job inputs to batch format
        job_inputs_reshape = torch.reshape(job_inputs, [batch_size, -1, self.job_input_dim])

        # (4) reshape gcn_outputs to batch format
        gcn_outputs_reshape = torch.reshape(gcn_outputs, [batch_size, -1, self.output_dim])

        # (5) reshape gsn_dag_summary to batch format
        gsn_dag_summ_reshape = gsn_dag_summary.view([batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = torch.unsqueeze(gsn_summ_backward_map, dim=0).tile(
            ([batch_size, 1, 1]))
        gsn_dag_summ_extend = torch.matmul(gsn_summ_backward_map_extend, gsn_dag_summ_reshape)

        # (6) reshape gsn_global_summary to batch format
        gsn_global_summ_reshape = gsn_global_summary.view([batch_size, -1, self.output_dim])
        gsn_global_summ_extend_job = gsn_global_summ_reshape.tile([1, gsn_dag_summ_reshape.size()[1], 1])
        gsn_global_summ_extend_node = gsn_global_summ_reshape.tile([1, gsn_dag_summ_extend.size()[1], 1])

        # (4) actor neural network
        # -- part A, the distribution over nodes --

        merge_node = torch.concat([
            node_inputs_reshape, gcn_outputs_reshape,
            gsn_dag_summ_extend,
            gsn_global_summ_extend_node], dim=2).to(torch.float32)

        # for nodes
        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        # reshape the output dimension (batch_size, total_num_nodes)
        node_outputs = torch.reshape(node_outputs, [batch_size, -1])

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        # do masked softmax over nodes on the graph
        node_outputs = self.softmax(node_outputs)

        # -- part B, the distribution over executor limits --

        merge_job = torch.concat([
            job_inputs_reshape,
            gsn_dag_summ_reshape,
            gsn_global_summ_extend_job], dim=2)

        expanded_state = expand_act_on_state(
            merge_job, [l / 50.0 for l in self.executor_levels])

        # for jobs
        y = self.fc5(expanded_state)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        job_outputs = self.fc4(y)

        # reshape the output dimension (batch_size, num_jobs * num_exec_limits)
        job_outputs = job_outputs.view([batch_size, -1])

        # valid mask on job
        job_valid_mask = (job_valid_mask - 1) * 10000.0

        # apply mask
        job_outputs = job_outputs + job_valid_mask

        # reshape output dimension for softmaxing the executor limits
        # (batch_size, num_jobs, num_exec_limits)
        job_outputs = job_outputs.view([batch_size, -1, len(self.executor_levels)])

        # do masked softmax over jobs
        job_outputs = self.softmax(job_outputs)

        return node_outputs, job_outputs


class ActorAgent(Agent):
    def __init__(self, node_input_dim, job_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps=1e-6, lr=0.002):

        Agent.__init__(self)

        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = nn.LeakyReLU()
        self.optimizer = torch.optim.Adam

        self.gcn = GraphCNN(node_input_dim, hid_dims, output_dim, max_depth)
        self.gsn = GraphSNN(node_input_dim + output_dim, hid_dims, output_dim)
        self.actor_network = ActorNetwork(node_input_dim, job_input_dim, output_dim, executor_levels)

        if cuda:
            self.gcn = self.gcn.cuda()
            self.gsn = self.gsn.cuda()
            self.actor_network = self.actor_network.cuda()
        if mps:
            self.gcn = self.gcn.device("mps")
            self.gsn = self.gsn.device("mps")
            self.actor_network = self.actor_network.device("mps")

        # for computing and storing message passing path
        self.postman = Postman()

        self.optimizer1 = self.optimizer(self.actor_network.parameters(), lr=lr)
        self.optimizer2 = self.optimizer(self.gsn.parameters(), lr=lr)
        self.optimizer3 = self.optimizer(self.gcn.parameters(), lr=lr)

    def set_params(self, params):
        self.gcn.load_state_dict(params[0])
        self.gsn.load_state_dict(params[1])
        self.actor_network.load_state_dict(params[2])

    def get_params(self):
        return [self.gcn.state_dict(), self.gsn.state_dict(), self.actor_network.state_dict()]

    def apply_gradients(self, loss):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer3.step()

    def save_model(self, file_path):
        print(f"save model {file_path}")
        torch.save(self.gcn.state_dict(), file_path + "gcn.pth")
        torch.save(self.gsn.state_dict(), file_path + "gsn.pth")
        torch.save(self.actor_network.state_dict(), file_path + "actor.pth")

    def get_params(self):
        return [self.gcn.state_dict(), self.gsn.state_dict(), self.actor_network.state_dict()]

    def set_params(self, params):
        self.gcn.load_state_dict(params[0])
        self.gsn.load_state_dict(params[1])
        self.actor_network.load_state_dict(params[2])

    def get_gradients(self, node_inputs, job_inputs,
                      node_valid_mask, job_valid_mask,
                      gcn_mats, gcn_masks, summ_mats,
                      running_dags_mat, dag_summ_backward_map,
                      node_act_vec, job_act_vec, adv, entropy_weight):

        gcn_output = self.gcn.forward(gcn_mats, gcn_masks, node_inputs)

        gsn_summaries = self.gsn.summarize(summ_mats, running_dags_mat, torch.concat([node_inputs, gcn_output], dim=1))

        node_act_probs, job_act_probs = self.actor_network.forward(
            node_inputs, gcn_output, job_inputs,
            gsn_summaries[0], gsn_summaries[1],
            node_valid_mask, job_valid_mask,
            dag_summ_backward_map)

        # select node action probability
        selected_node_prob = torch.sum(torch.multiply(
            node_act_probs, node_act_vec), dim=1, keepdim=True)

        # select job action probability
        selected_job_prob = torch.sum(torch.sum(torch.multiply(
            job_act_probs, job_act_vec), dim=2), dim=1, keepdim=True)

        # actor loss due to advantge (negated)
        adv_loss = torch.sum(torch.multiply(
            torch.log(selected_node_prob * selected_job_prob + \
                      self.eps), -adv))

        # node_entropy
        node_entropy = torch.sum(torch.multiply(
            node_act_probs, torch.log(node_act_probs + self.eps)))

        # prob on each job

        prob_each_job = torch.reshape(
            torch.sparse.mm(summ_mats, torch.reshape(node_act_probs, [-1, 1])),
            [node_act_probs.size()[0], -1])

        # job entropy
        job_entropy = torch.sum(torch.multiply(prob_each_job, torch.sum(torch.multiply(job_act_probs,
                                                                                       torch.log(
                                                                                           job_act_probs + self.eps)),
                                                                        dim=2)))

        # entropy loss
        entropy_loss = node_entropy + job_entropy

        # normalize entropy
        entropy_loss /= \
            (torch.log(torch.tensor(node_act_probs.size()[1]).to(torch.float32))) + \
            torch.log(torch.tensor(float(len(self.executor_levels))))
        # normalize over batch size (note: adv_loss is sum)
        # * tf.cast(tf.shape(self.node_act_probs)[0], tf.float32)

        # define combined loss
        act_loss = adv_loss + entropy_weight * entropy_loss

        return act_loss, [adv_loss, entropy_loss]

    def predict(self, node_inputs, job_inputs,
                node_valid_mask, job_valid_mask,
                gcn_mats, gcn_masks, summ_mats,
                running_dags_mat, dag_summ_backward_map):

        with torch.no_grad():
            gcn_output = self.gcn.forward(gcn_mats, gcn_masks, node_inputs)
            gsn_summaries = self.gsn.summarize(summ_mats, running_dags_mat,
                                               torch.concat([node_inputs, gcn_output], dim=1))
            node_act_probs, job_act_probs = self.actor_network.forward(
                node_inputs, gcn_output, job_inputs,
                gsn_summaries[0], gsn_summaries[1],
                node_valid_mask, job_valid_mask,
                dag_summ_backward_map)

        # draw action based on the probability (from OpenAI baselines)
        # node_acts [batch_size, 1]
        logits = torch.log(node_act_probs)
        noise = torch.rand(logits.size())
        if cuda:
            noise = noise.cuda()

        # if cuda:
        #     logits = logits.cuda()
        #     noise = noise.cuda()
        node_acts = torch.argmax(logits - torch.log(-torch.log(noise)), 1)

        # job_acts [batch_size, num_jobs, 1]
        logits = torch.log(job_act_probs)
        noise = torch.rand(logits.size())
        if cuda:
            noise = noise.cuda()
        # if cuda:
        #     logits = logits.cuda()
        #     noise = noise.cuda()
        job_acts = torch.argmax(logits - torch.log(-torch.log(noise)), 2)

        return node_act_probs, job_act_probs, node_acts, job_acts

    def translate_state(self, obs):
        """
        Translate the observation to matrix form
        """
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        # compute total number of nodes
        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        job_inputs = np.zeros([len(job_dags), self.job_input_dim])

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += exec_commit.commit[s][n]

        # gather job level inputs
        job_idx = 0
        for job_dag in job_dags:
            # number of executors in the job
            job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
            # the current executor belongs to this job or not
            if job_dag is source_job:
                job_inputs[job_idx, 1] = 2
            else:
                job_inputs[job_idx, 1] = -2
            # number of source executors
            job_inputs[job_idx, 2] = num_source_exec / 20.0

            job_idx += 1

        # gather node level inputs
        node_idx = 0
        job_idx = 0
        for job_dag in job_dags:
            for node in job_dag.nodes:
                # copy the feature from job_input first
                node_inputs[node_idx, :3] = job_inputs[job_idx, :3]

                # work on the node
                node_inputs[node_idx, 3] = \
                    (node.num_tasks - node.next_task_idx) * \
                    node.tasks[-1].duration / 100000.0

                # number of tasks left
                node_inputs[node_idx, 4] = \
                    (node.num_tasks - node.next_task_idx) / 200.0

                node_idx += 1

            job_idx += 1

        return node_inputs, job_inputs, \
               job_dags, source_job, num_source_exec, \
               frontier_nodes, executor_limits, \
               exec_commit, moving_executors, \
               exec_map, action_map

    def get_valid_masks(self, job_dags, frontier_nodes,
                        source_job, num_source_exec, exec_map, action_map):

        job_valid_mask = np.zeros([1, len(job_dags) * len(self.executor_levels)])

        job_valid = {}  # if job is saturated, don't assign node

        base = 0
        for job_dag in job_dags:
            # new executor level depends on the source of executor
            if job_dag is source_job:
                least_exec_amount = \
                    exec_map[job_dag] - num_source_exec + 1
                # +1 because we want at least one executor
                # for this job
            else:
                least_exec_amount = exec_map[job_dag] + 1
                # +1 because of the same reason above

            assert least_exec_amount > 0
            assert least_exec_amount <= self.executor_levels[-1] + 1

            # find the index for first valid executor limit
            exec_level_idx = bisect.bisect_left(
                self.executor_levels, least_exec_amount)

            if exec_level_idx >= len(self.executor_levels):
                job_valid[job_dag] = False
            else:
                job_valid[job_dag] = True

            for l in range(exec_level_idx, len(self.executor_levels)):
                job_valid_mask[0, base + l] = 1

            base += self.executor_levels[-1]

        total_num_nodes = int(np.sum(
            job_dag.num_nodes for job_dag in job_dags))

        node_valid_mask = np.zeros([1, total_num_nodes])

        for node in frontier_nodes:
            if job_valid[node.job_dag]:
                act = action_map.inverse_map[node]
                node_valid_mask[0, act] = 1

        return node_valid_mask, job_valid_mask

    def invoke_model(self, obs):

        # implement this module here for training
        # (to pick up state and action to record)
        node_inputs, job_inputs, \
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, \
        exec_map, action_map = self.translate_state(obs)

        # get message passing path (with cache)
        gcn_mats, gcn_masks, dag_summ_backward_map, \
        running_dags_mat, job_dags_changed = \
            self.postman.get_msg_path(job_dags)

        # get node and job valid masks
        node_valid_mask, job_valid_mask = \
            self.get_valid_masks(job_dags, frontier_nodes,
                                 source_job, num_source_exec, exec_map, action_map)

        gcn_mats = torch.stack(gcn_mats)

        # get summarization path that ignores finished nodes
        summ_mats = get_unfinished_nodes_summ_mat(job_dags)

        node_inputs = torch.Tensor(node_inputs).to(torch.float32)
        job_inputs = torch.tensor(job_inputs).to(torch.float32)
        node_valid_mask = torch.tensor(node_valid_mask).to(torch.float32)
        job_valid_mask = torch.tensor(job_valid_mask).to(torch.float32)
        gcn_masks = torch.tensor(np.array(gcn_masks)).to(torch.float32)
        dag_summ_backward_map = torch.Tensor(dag_summ_backward_map).to(torch.float32)

        if cuda:
            node_inputs = node_inputs.cuda()
            job_inputs = job_inputs.cuda()
            node_valid_mask = node_valid_mask.cuda()
            job_valid_mask = job_valid_mask.cuda()
            gcn_mats = gcn_mats.cuda()
            gcn_masks = gcn_masks.cuda()
            summ_mats = summ_mats.cuda()
            running_dags_mat = running_dags_mat.cuda()
            dag_summ_backward_map = dag_summ_backward_map.cuda()

        # invoke learning model
        node_act_probs, job_act_probs, node_acts, job_acts = self.predict(node_inputs, job_inputs,
                                                                          node_valid_mask, job_valid_mask,
                                                                          gcn_mats, gcn_masks, summ_mats,
                                                                          running_dags_mat, dag_summ_backward_map)

        return node_acts, job_acts, \
               node_act_probs, job_act_probs, \
               node_inputs, job_inputs, \
               node_valid_mask, job_valid_mask, \
               gcn_mats, gcn_masks, summ_mats, \
               running_dags_mat, dag_summ_backward_map, \
               exec_map, job_dags_changed

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs

        if len(frontier_nodes) == 0:
            # no action to take
            return None, num_source_exec

        # invoking the learning model
        node_act, job_act, \
        node_act_probs, job_act_probs, \
        node_inputs, job_inputs, \
        node_valid_mask, job_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_dags_mat, dag_summ_backward_map, \
        exec_map, job_dags_changed = self.invoke_model(obs)

        if sum(node_valid_mask[0, :]) == 0:
            # no node is valid to assign
            return None, num_source_exec

        # node_act should be valid
        try:
            assert node_valid_mask[0, node_act[0]] == 1
        except:
            return None, num_source_exec

        # parse node action
        node = action_map[node_act[0].item()]

        # find job index based on node
        job_idx = job_dags.index(node.job_dag)

        # job_act should be valid
        assert job_valid_mask[0, job_act[0, job_idx] + len(self.executor_levels) * job_idx] == 1

        # find out the executor limit decision
        if node.job_dag is source_job:
            agent_exec_act = self.executor_levels[
                                 job_act[0, job_idx]] - \
                             exec_map[node.job_dag] + \
                             num_source_exec
        else:
            agent_exec_act = self.executor_levels[
                                 job_act[0, job_idx]] - exec_map[node.job_dag]

        # parse job limit action
        use_exec = min(
            node.num_tasks - node.next_task_idx - \
            exec_commit.node_commit[node] - \
            moving_executors.count(node),
            agent_exec_act, num_source_exec)

        return node, use_exec

    def restore_models(self, file_path):
        print("restore models " + file_path)
        if not cuda:
            self.gcn.load_state_dict(torch.load(file_path + "gcn.pth"))
            self.gsn.load_state_dict(torch.load(file_path + "gsn.pth"))
            self.actor_network.load_state_dict(torch.load(file_path + "actor.pth"))
        else:
            self.gcn.load_state_dict(torch.load(file_path + "gcn.pth", map_location='cuda'))
            self.gsn.load_state_dict(torch.load(file_path + "gsn.pth", map_location='cuda'))
            self.actor_network.load_state_dict(torch.load(file_path + "actor.pth", map_location='cuda'))

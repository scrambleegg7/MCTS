
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import operator
import networkx as nx
import copy
import matplotlib.pyplot as plt

from GameState3 import GameState

from my_moduler import get_module_logger

mylogger = get_module_logger(__name__)

class Policy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def move(self, state):
        pass


class RandomPolicy(Policy):
    def move(self, state):
        """Chooses moves randomly from the legal moves in a given state"""
        legal_moves = state.legal_moves()
        #print("<RandomPolicy> legal_moves : %s" % legal_moves)
        idx = np.random.randint(len(legal_moves))
        return legal_moves[idx]
# return np.random.choice(state.legal_moves())

# avoid to divide by 0
#
EPSILON = 10e-6

class MCTSPolicy(Policy):

    def __init__(self, player="X"):
        """
        Implementation of Monte Carlo Tree Search
        Creates a root of an MCTS tree to keep track of the information
        obtained throughout the course of the game in the form of a tree
        of MCTS nodes
        The data structure of a node consists of:
          - the game state which it corresponds to
          - w, the number of wins that have occurred at or below it in the tree
          - n, the number of plays that have occurred at or below it in the tree
          - expanded, whether all the children (legal moves) of the node have
            been added to the tree
        To access the node attributes, use the following format. For example,
        to access the attribute 'n' of the root node:
          policy = MCTSPolicy()
          current_node = policy.root
          policy.tree.node[current_node]['n']
        """
        self.digraph = nx.DiGraph()
        self.player = player
        self.num_simulations = 0
        # Constant parameter to weight exploration vs. exploitation for UCT
        self.uct_c = 1. / np.sqrt(2)

        self.node_counter = 0

        empty_board = GameState()
        self.digraph.add_node(self.node_counter,w=0,vn=0,uct=0, expanded=False, state=empty_board)
        empty_board_node_id = self.node_counter
        self.node_counter += 1

        self.last_move = None

        #if player is 'O': # first hand of board
        #for successor in [empty_board.transition_function(move) for move in empty_board.legal_moves()]:
        #    self.digraph.add_node(self.node_counter, w=0,vn = 0,uct = 0, expanded = False, state = successor)
        #    self.digraph.add_edge(empty_board_node_id, self.node_counter)
        #    self.node_counter += 1

    def drawGraph(self):
        #nx.draw_networkx(self.digraph)
        #plt.show()

        pos = nx.spring_layout(self.digraph,scale=3)

        nx.draw(self.digraph, pos)
        plt.show()

    def move(self,starting_state):

        best_move = self.uctsearch(starting_state)
        return best_move

    def uctsearch(self,starting_state):

        starting_node = None
        starting_state = copy.deepcopy(starting_state)

        found = False
        for n in self.digraph.nodes():
            if self.digraph.node[n]["state"] == starting_state:
                mylogger.debug("matched existing node id -> %d ", n)
                starting_node = n
                found = True
        if not found:
            self.digraph.add_node(self.node_counter, w=0,vn=0,uct=0, expanded=False, state = starting_state)
            mylogger.debug("root is not in digraph. add node. %d " , self.node_counter)
            starting_node = self.node_counter
            self.node_counter += 1

        for i in range(100):
            node = self.treepolicy(starting_node)
            reward = self.defaultpolicy(node)

            #mylogger.debug("reward", reward)
            self.backup(node,reward)

        best_child_id, move = self.best(starting_node)

        mylogger.debug("best Child id --> %d",best_child_id)
        mylogger.debug("best move --> %d", move)
        return move


    def treepolicy(self,node):

        # node -- Basically, it is root node

        state = self.digraph.node[node]['state']
        children = self.digraph.successors(node)
        length_of_children = len( list( children) )
        #mylogger.debug("any winnder ?", state.winner())
        #mylogger.debug("length of children under node ", length_of_children)

        #available_moves = state.legal_moves()
        #mylogger.debug("available moves.", available_moves)

        while state.winner() == None:

            #print("<Treepolicy> Node #", node)
            #print(state)


            if length_of_children == 0:
                return self.expand(node)
            if length_of_children > 0:

                if random.uniform(0,1)<.5:
                    #print("**** best node randomly pick up **** < 50% ")
                    node, _ = self.best(node)
                    #return node
                else:
                    if self.fully_expanded(node) == False:
                        return self.expand(node)
                    else:
                        #print("**** best node randomly pick up (case of fully expanded )**** > 50% ")
                        node, _ = self.best(node)
                        #return node

                state = self.digraph.node[node]['state']
                children = self.digraph.successors(node)
                length_of_children = len( list( children) )

        return node

    def treepolicy2(self,node):

        # node -- Basically, it is root node

        state = self.digraph.node[node]['state']
        children = self.digraph.successors(node)
        length_of_children = len( list( children) )
        #print("Node #", node)
        #print("any winnder ?", state.winner())
        #print("length of children under node ", length_of_children)

        #available_moves = state.legal_moves()
        ##print("available moves.", available_moves)

        if length_of_children == 0:
            return self.expand(node)
        if length_of_children > 0:

            if random.uniform(0,1)<.5:
                node, _ = self.best(node)
                return node
            else:
                if self.fully_expanded(node) == False:
                    return self.expand(node)
                else:
                    node, _ = self.best(node)
                    return node


    def treepolicy3(self,root):

        if root not in self.digraph.nodes():
            #print("root is not in digraph. add node.")
            self.digraph.add_node(self.node_counter, w=0,vn=0,uct=0, expanded=False, state = root)
            self.node_counter += 1
            return root
        elif not self.digraph.node[root]['expanded']:
            #print('root in digraph but not expanded')
            return root  # This is the node to expand
        else:
            #print('root expanded, move on to a child')
            # Handle the general case
            children = self.digraph.successors(root)
            uct_values = {}
            for child_node in children:
                uct_values[child_node] = self.uct(state=child_node)

            # Choose the child node that maximizes the expected value given by UCT
            best_child_node = max(uct_values.items(), key=operator.itemgetter(1))[0]

        return self.treepolicy(best_child_node)

    def defaultpolicy(self, node):
        """
        Conducts a light playout from the specified node
        :return: The reward obtained once a terminal state is reached
        """
        random_policy = RandomPolicy()
        current_state = self.digraph.node[node]['state']
        while not current_state.winner():
            move = random_policy.move(current_state)
            current_state = current_state.transition_function(move)


        #print("-"*20)
        #print("<DEFAULTPOLICY> result:")
        #print(current_state)
        #print("-"*20)

        if current_state.winner() == self.player:
            return 1
        else:
            return 0

    def fully_expanded(self,node):

        current_state = self.digraph.node[node]["state"]
        current_legal_moves = current_state.legal_moves()
        children = self.digraph.successors(node)

        move_list = []
        for c in list(children):
            action = self.digraph.get_edge_data(node, c)['action']
            if action in current_legal_moves:
                move_list.append(action)

        if len(move_list) == len(current_legal_moves):
            return True
        return False


    def expand(self,node):

        #print("-" * 20 )
        #print("EXPAND LOGIC under node:", node)
        #print("-" * 20 )

        children_moves = []
        # As long as this node has at least one unvisited child, choose a legal move
        children = self.digraph.successors(node)
        for c in list(children):
            action = self.digraph.get_edge_data(node, c)['action']
            children_moves.append( action  )
        #print("children moves in the past", children_moves)

        node_current_state = self.digraph.node[node]['state']
        node_legal_moves = node_current_state.legal_moves()
        available_moves = [m for m in node_legal_moves if m not in children_moves]

        move = np.random.choice(available_moves)

        #print("selected move", move)
        child = self.digraph.node[node]['state'].transition_function(move)


        #print("node append Parent:[%d] --> Child:[%d]" % (node, self.node_counter))
        self.digraph.add_node(self.node_counter ,w=0, vn=0, uct=0, expanded=False, state=child)
        self.digraph.add_edge(node, self.node_counter, action = move)
        child_node_id = self.node_counter
        self.node_counter += 1

        return child_node_id

    def best(self,node):

        #print("<BEST> search best child node under parent node %d" % node)

        children = list( self.digraph.successors(node) )
        uct_values = {}
        for child_node in children:
            uct_values[child_node] = self.uct(child_node,node)

        uct_values_list = np.array( list( uct_values.values()  ) )
        max_uct_value = np.max(uct_values_list)
        max_list_index = np.where( uct_values_list == max_uct_value)[0]

        ##print("uct_values_list", uct_values_list)
        ##print("max_uct_value", max_uct_value)
        ##print("max_list_index", max_list_index)
        if len(max_list_index) > 0:
            max_list_index = np.random.choice(max_list_index)

        max_child_id = list( uct_values.keys() )[max_list_index]


        action = self.digraph.get_edge_data(node, max_child_id)['action']

        #print("-" * 20)
        #print("<BEST> best child id : %d  move : %d"  % (max_child_id, action))
        #print("-" * 20)

        return max_child_id, action


    def uct(self,state,node):
        """
        Returns the expected value of a state, calculated as a weighted sum of
        its exploitation value and exploration value
        """
        overall_n = self.digraph.node[node]['vn']
        n = self.digraph.node[state]['vn']  # Number of plays from this node
        w = self.digraph.node[state]['w']  # Number of wins from this node
        t = self.num_simulations
        c = self.uct_c
        epsilon = EPSILON

        exploitation_value = w / (n + epsilon)
        exploration_value = 2. * c * np.sqrt(np.log(overall_n) / (n + epsilon))
        #print('<UCT> exploration_value: {}'.format(exploration_value))

        value = exploitation_value + exploration_value

        #print('<UCT> UCT value {:.3f} for child_id : {}'.format(value, state))

        self.digraph.node[state]['uct'] = value

        return value

    def backup(self, last_visited, reward):
        """
        Walk the path upwards to the root, incrementing the
        'n' and 'w' attributes of the nodes along the way
        """
        current = last_visited
        while True:
            self.digraph.node[current]['vn'] += 1
            self.digraph.node[current]['w'] += reward

            #print("<BACKUP> node ID  ", current)
            #print('<BACKUP> Updating to n={} and w={}:\n{}'.format(self.digraph.node[current]['vn'],
            #                                              self.digraph.node[current]['w'],
            #                                              self.digraph.node[current]['state']))

            # Terminate when we reach the empty board
            if self.digraph.node[current]['state'] == GameState():
                break
            # Todo:
            # Does this handle the necessary termination conditions for both 'X' and 'O'?
            # As far as we can tell, it does

            # Will throw an IndexError when we arrive at a node with no predecessors
            # Todo: see if this additional check is no longer necessary
            try:
                current = list(self.digraph.predecessors(current))[0]
            except IndexError:
                break

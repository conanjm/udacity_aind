
from sample_players import DataPlayer
from mcts import *
from alpha_beta_search import *
import random

class CustomPlayer_AlphaBeta(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 4:
            if state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            ###### iterative deepening ######
            depth_limit = 5
            for depth in range(1, depth_limit + 1):
                best_move = alpha_beta_search(state, self.player_id, depth)
            self.queue.put(best_move)

class CustomPlayer_MCTS(DataPlayer):
    """
    Implement an agent to play knight's Isolation with Monte Carlo Tree Search
    """
    def mcts(self, state):
        root = MCTS_Node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        for _ in range(iter_limit):
            child = tree_policy(root)
            if not child:
                continue
            reward = default_policy(child.state)
            backup(child, reward)

        idx = root.children.index(best_child(root))
        return root.children_actions[idx]

    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state))

CustomPlayer = CustomPlayer_AlphaBeta
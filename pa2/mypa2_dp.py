from collections import defaultdict
from mymdp import MDP
import math


class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need to append the v table to this list. (include the initial value)
    """

    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.            
        """
        self.q = dict()
        self.v = dict()
        self.pi = dict()
        self.mdp = mdp
        self.thresh = conv_thresh
        self.v_update_history = list()

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """
        for s in self.mdp.states():
            num_actions = len(self.mdp.actions(s))
            for a in self.mdp.actions(s):
                if s not in self.pi:
                    self.pi[s] = {}
                self.pi[s][a] = 1 / num_actions

    def computeq_fromv(self, v: dict[str, float]) -> dict[str, dict[str, float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """

        # Initialize q(s,a) := 0 for all state-action pairs
        q = {state: {action: 0 for action in self.mdp.actions(state)}
             for state in self.mdp.states()}

        # Iterate over every state,action,sstate combination
        for state in self.mdp.states():
            for action in self.mdp.actions(state):
                for sstate, transit_prob in self.mdp.T(state, action):

                    r = self.mdp.R(state, action, sstate)
                    gamma = self.mdp.gamma

                    # Q(s,a) += p(s', r | s, a) * (r + yV(s'))
                    q[state][action] += transit_prob * (r + gamma * v[sstate])
        return q

    def greedy_policy_improvement(self, v: dict[str, float]) -> dict[str, dict[str, float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """
        q = self.computeq_fromv(v)
        pi = {state: {} for state in self.mdp.states()}

        for state in q:
            greedy_action = max(q[state], key=q[state].get)
            pi[state] = {greedy_action: 1}
        return pi

    def check_term(self, v: dict[str, float], next_v: dict[str, float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows: 
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        for state in v:
            if abs(next_v[state] - v[state]) > self.thresh:
                return True
        return False


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """

    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """
        super().__init__(mdp, conv_thresh)
        super().init_random_policy()  # initialize its policy function with the random policy

    def __iter_policy_eval(self, pi: dict[str, dict[str, float]]) -> dict[str, float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        delta = 1
        while delta > self.thresh:
            delta = 0
            v = {state: 0 for state in self.mdp.states()}
            for state in self.v:
                initial_state_value = self.v[state]
                # self.v[state] = self.q[state][max(pi[state])] #kinda hacky, don't do max(pi)

                # self.v[state] = 0
                for action in pi[state]:
                    policy_prob = pi[state][action]

                    for sstate, transit_prob in self.mdp.T(state, action):
                        r = self.mdp.R(state, action, sstate)
                        gamma = self.mdp.gamma

                        v[state] += policy_prob * transit_prob*(r + gamma*self.v[sstate])

                delta = max(delta, abs(v[state] - initial_state_value))

            self.v_update_history.append(self.v)
            self.v = v

        return self.v

    def policy_iteration(self) -> dict[str, dict[str, float]]:
        """Policy iteration algorithm. Iterating iter_policy_eval and greedy_policy_improvement, update the policy pi until convergence of the state-value function.

        This function is called to run PI. 
        e.g.
        mdp = MDP("./mdp1.json")
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        # 1. Initialization
        self.v = {state: 0 for state in self.mdp.states()}
        self.q = self.computeq_fromv(self.v)
        self.init_random_policy()

        # 3. Policy Improvement
        policy_stable = False
        while not policy_stable:
            # Assume Policy is Stable
            policy_stable = True

            self.v = self.__iter_policy_eval(self.pi)

            for state in self.mdp.states():
                old_action = max(self.pi[state], key=self.pi[state].get)
                new_action = max(self.q[state], key=self.q[state].get)
                self.pi[state] = {new_action: 1}

                # A single change to the policy means it is NOT stable
                if old_action != new_action:
                    policy_stable = False

        return self.pi



class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """

    def __init__(self, mdp: MDP, conv_thresh: float = 0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """
        super().__init__(mdp, conv_thresh)
        super().init_random_policy()  # initialize its policy function with the random policy

    def value_iteration(self) -> dict[str, dict[str, float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration. After that, generate the corresponding optimal policy pi.

        This function is called to run VI. 
        e.g.
        mdp = MDP("./mdp1.json")
        via = VIAgent(mdp)
        via.value_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        self.v = {state: 0 for state in self.mdp.states()}
        delta = 1

        while delta > self.thresh:
            delta = 0
            v = {state: 0 for state in self.mdp.states()}
            for state in self.mdp.states():
                initial_state_value = self.v[state]

                # Select the maximum Q-value across all actions
                self.q = self.computeq_fromv(self.v)
                v[state] = max(self.q[state].values())

                delta = max(delta, abs(v[state] - initial_state_value))

            self.v_update_history.append(self.v)
            self.v = v

        return self.greedy_policy_improvement(self.v)


if __name__ == "__main__":
    mdp = MDP("./mdp1.json")
    test_pi = False

    #Test PI
    if test_pi:
        pia = PIAgent(mdp)
        pi_p = pia.policy_iteration()

        for v in pia.v_update_history:
            print(v)
        print(f'Policy iteration pi: {pi_p}')

    #Test VI
    else:
        via = VIAgent(mdp)
        vi_v = via.value_iteration()

        for v in via.v_update_history:
            print(v)
        print(f'Value iteration pi: {vi_v}')




# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # go through each iteration
        for i in range(self.iterations):
            # get all possible states
            states = self.mdp.getStates()
            cur_values = self.values.copy()

            # go though each state
            for i in range(len(states)):
                # get current state
                cur_state = states[i]

                # get all possible actions for current state
                cur_actions = self.mdp.getPossibleActions(cur_state)
                q_values = util.Counter()

                # determine if current state is not a terminal
                if not self.mdp.isTerminal(cur_state):
                    # determine q value for each action
                    for action in cur_actions:
                        q_values[action] = self.computeQValueFromValues(cur_state, action)
                    # get the maximum q value
                    cur_values[cur_state] = max(q_values.values())
            
            # copy cur_values into self.values
            self.values = cur_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # get all transitions
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        
        # go through each transition, and calculate q value
        for i in range(len(transitions)):
            nextState = transitions[i][0]
            prob = transitions[i][1]
            reward = self.mdp.getReward(state, action, nextState)
            discount = self.discount
            value = self.values[nextState]
            q_value += prob * (reward + discount * value)

        return q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get all possible actions for state
        actions = self.mdp.getPossibleActions(state)

        # determine if state is a terminal
        if self.mdp.isTerminal(state):
            return None

        q_values = util.Counter()

        # go through each action
        for action in actions:
            # compute the q value for each action
            q_values[action] = self.computeQValueFromValues(state, action)

        return q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # go through each iteration
        for i in range(self.iterations):
            # get all states
            states = self.mdp.getStates()
            num_states = len(states)
            cur_values = self.values.copy()

            # get current state
            cur_state = states[i % num_states]

            # get all possible actions for current state
            cur_actions = self.mdp.getPossibleActions(cur_state)
            q_values = util.Counter()

            # determine if current state is not a terminal
            if not self.mdp.isTerminal(cur_state):
                # go through each action
                for action in cur_actions:
                    # compute the q value for each action
                    q_values[action] = self.computeQValueFromValues(cur_state, action)

                # get the maximum q value
                cur_values[cur_state] = max(q_values.values())

            # copy cur_values into self.values
            self.values = cur_values

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # compute all predecessors
        predecessors = {}
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for transition in transitions:
                        nextState = transition[0]
                        prob = transition[1]
                        if prob != 0:
                            if nextState in predecessors:
                                predecessors[nextState].add(state)
                            else:
                                predecessors[nextState] = {state}

        # initialize empty priority queue
        pqueue = util.PriorityQueue()
        
        # iterate through each state
        for s in states:
            s_qvalues = util.Counter()

            # determine if state is a terminal or not
            if not self.mdp.isTerminal(s):
                    # get possible actions for current state
                    actions = self.mdp.getPossibleActions(s)

                    # determine each q value for each action
                    for action in actions:
                        s_qvalues[action] = self.computeQValueFromValues(s, action)
                    
                    # find the maximum q value
                    max_s_qvalue = max(s_qvalues.values())

                    # find the absolute difference between current value of s in self.values and highest Q-value across all possible actions from s
                    diff = abs(max_s_qvalue - self.values[s])

                    # push s into priority queue with priority -diff
                    pqueue.update(s, -diff)
        
        # go through each iteration
        for i in range(self.iterations):
            # if priority queue is empty, then terminate
            if pqueue.isEmpty():
                break

            # pop a state s off the priority queue
            s = pqueue.pop()
            s_qvalues = util.Counter()

            # determine if state is a terminal or not
            if not self.mdp.isTerminal(s):
                # get possible actions for current state
                actions = self.mdp.getPossibleActions(s)

                # determine each q value for each action
                for action in actions:
                    s_qvalues[action] = self.computeQValueFromValues(s, action)

                # find the maximum q value
                max_s_qvalue = max(s_qvalues.values())

                # update s's value in self.values
                self.values[s] = max_s_qvalue
            
            # go through each predecessor p of s
            for p in predecessors[s]:
                # get all possible actions for p
                actions = self.mdp.getPossibleActions(p)
                p_qvalues = util.Counter()

                # determine each q value for each action 
                for action in actions:
                    p_qvalues[action] = self.computeQValueFromValues(p, action)

                # find the maximum q value
                max_p_qvalue = max(p_qvalues.values())

                # find absolute difference between current value of p in self.values and highest q-value across all possible actions from p
                diff = abs(self.values[p] - max_p_qvalue)

                # determine if diff is greater than theta
                if diff > self.theta:
                    # push p into the priority queue with priority -diff
                    pqueue.update(p, -diff)

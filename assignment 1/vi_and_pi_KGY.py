### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary|
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    n_iter = 0
    while True:
        prev_value_function = np.copy(value_function)
        for s in range(nS):
            action = policy[s]
            temp_v = 0
            ### Stochastic 에서 확률적으로 같은 action을 하더라도 다른 state로 넘어가는 경우가 있음####
            for state_action_pair in P[s][action]:
                prob, next_state, reward, terminal = state_action_pair
                # print(prob, next_state, reward, terminal)
                stochastic_v =  prob * (reward + gamma * prev_value_function[next_state])
                temp_v += stochastic_v
            value_function[s] = temp_v
        n_iter +=1
        diff = np.max(np.abs(value_function - prev_value_function))
        print(diff)
        if diff < tol:
            break;
    ############################
    return value_function

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')
    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(nS):
        q_max = 0
        q_list = np.array([])
        for a in range(nA):
            temp_q = 0
            for state_action_pair in P[s][a]:
                prob, next_state, reward, terminal = state_action_pair
                stochastic_q =  prob * (reward + gamma * value_from_policy[next_state]) 
                temp_q += stochastic_q
            q_list = np.append(q_list, temp_q)
        q_max = np.argmax(q_list)
        new_policy[s] = q_max
#         print(new_policy)
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    n_iter = 0
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ###########################YOUR IMPLEMENTATION HERE #
    
    prev_value_function = np.copy(value_function)
    while True :
        prev_value_function = policy_evaluation(P, nS, nA, policy)
        policy = policy_improvement(P, nS, nA, prev_value_function, policy)
        value_function = policy_evaluation(P, nS, nA, policy)
        if np.sum(value_function - prev_value_function) <= 0:
            break
        n_iter+=1
        
    ############################
    return value_function, policy , n_iter

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    prev_value_function = np.copy(value_function)
    n_it = 1
    while True:
        for s in range(nS):
            v_list = np.array([])
            for a in range(nA):
                temp_v = 0
                ### Stochastic 에서 확률적으로 같은 action을 하더라도 다른 state로 넘어가는 경우가 있음####
                for state_action_pair in P[s][a]:
                    prob, next_state, reward, terminal = state_action_pair
                    stochastic_v =  prob * (reward + gamma * prev_value_function[next_state])
                    temp_v += stochastic_v
                v_list = np.append(v_list, temp_v)
            value_function[s] = np.max(v_list)
            policy[s] = np.argmax(v_list)
        
        diff = np.max(np.abs(value_function - prev_value_function))
        n_it +=1
        prev_value_function = np.copy(value_function)
        if diff < tol:
            break;
            
    
    ############################
    return value_function, policy, n_it

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
      print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    # env = gym.make("Deterministic-4x4-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi, n_iter= policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    # render_single(env, p_pi, 100)
    print("\n Total Iteration : %d \n"%(n_iter))
    print(V_pi)
    print(p_pi)

    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

    V_vi, p_vi, n_it = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    
    # render_single(env, p_vi, 100)
    print("\n Total Iteration : %d \n"%(n_it))
    print(V_vi)
    print(p_vi)
    
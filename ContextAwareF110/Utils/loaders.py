from ContextAwareF110.Planners.PurePursuit import PurePursuit
from ContextAwareF110.Planners.FollowTheGap import FTG
from ContextAwareF110.Planners.TD3Planners import TD3Trainer, TD3Tester
from ContextAwareF110.Planners.SACPlanners import SACTrainer, SACTester
from ContextAwareF110.Planners.DreamerV2Planners import DreamerV2Trainer, DreamerV2Tester
from ContextAwareF110.Planners.DreamerV3Planners import DreamerV3Trainer, DreamerV3Tester
from ContextAwareF110.Planners.cDreamerPlanners import cDreamerTrainer, cDreamerTester
from ContextAwareF110.Planners.cbDreamerPlanners import cbDreamerTrainer, cbDreamerTester

from ContextAwareF110.Utils.RewardSignals import *


# TODO move to utils
def select_reward_function(run, conf, std_track):
    reward = run.reward
    if reward == "Progress":
        reward_function = ProgressReward(std_track)
    elif reward == "Cth": 
        reward_function = CrossTrackHeadReward(std_track, conf)
    elif reward == "TAL":
        reward_function = TALearningReward(conf, run)
    else: raise Exception("Unknown reward function: " + reward)
        
    return reward_function

# TODO move to utils
def select_agent(run, conf, architecture, train=True, init=False, context_info=[0.0, 0.0]):
    agent_type = architecture if architecture is not None else "TD3"
    if agent_type == "PP":
        agent = PurePursuit(run, conf, init=init, context_info=context_info) 
    elif agent_type == "FTG":
        agent = FTG(run, conf, context_info=context_info)
    elif agent_type == "TD3":
        agent = TD3Trainer(run, conf, init=init) if train else TD3Tester(run, conf)
    elif agent_type == "SAC":
        agent = SACTrainer(run, conf, init=init) if train else SACTester(run, conf)
    elif agent_type == "DreamerV2":
        agent = DreamerV2Trainer(run, conf) if train else DreamerV2Tester(run, conf)
    elif agent_type == "DreamerV3":
        agent = DreamerV3Trainer(run, conf, init=init) if train else DreamerV3Tester(run, conf)
    elif agent_type == "cDreamer":
        agent = cDreamerTrainer(run, conf, init=init) if train else cDreamerTester(run, conf)
    elif agent_type == "cbDreamer":
        agent = cbDreamerTrainer(run, conf, init=init) if train else cbDreamerTester(run, conf)
    else: raise Exception("Unknown agent type: " + agent_type)

    return agent
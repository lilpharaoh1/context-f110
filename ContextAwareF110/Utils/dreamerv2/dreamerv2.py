import numpy as np
import torch 
import torch.optim as optim
import os 

from ContextAwareF110.Utils.dreamerv2.training.config import MultiAgent, SingleAgent

from ContextAwareF110.Utils.dreamerv2.utils.module import get_parameters, FreezeParameters
from ContextAwareF110.Utils.dreamerv2.utils.algorithm import compute_return

from ContextAwareF110.Utils.dreamerv2.models.actor import DiscreteActionModel
from ContextAwareF110.Utils.dreamerv2.models.dense import DenseModel
from ContextAwareF110.Utils.dreamerv2.models.rssm import RSSM
from ContextAwareF110.Utils.dreamerv2.models.enc_dec import ObsDecoder, ObsEncoder
from ContextAwareF110.Utils.dreamerv2.utils.buffer import TransitionBuffer

class DreamerV2(object):
    def __init__(self, state_dim, action_dim, name, max_action=1, window_in=1, window_out=1, multiagent=True):
        self.name = name
        self.state_dim = state_dim
        self.max_action = max_action
        self.act_dim = action_dim
        self.window_in = window_in
        self.window_out = window_out
        self.state_buff = None

        # Load HP for training
        config = MultiAgent() if multiagent else SingleAgent()
        self.config = config
        self.kl_info = config.kl # used to compute kl_loss (???)
        self.seq_len = config.seq_len # Same as window_in
        self.horizon = config.horizon # Same as window_out
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals # Same as POLICY_FREQ
        self.seed_steps = config.seed_steps # How many episodes before training, we use window_in * BATCH_SIZE
        self.discount = config.discount_ # Same as GAMMA
        self.lambda_ = config.lambda_ # Used in compute_returns (???)
        self.loss_scale = config.loss_scale # How much to weigh loss components (kl, reward, discount)
        self.actor_entropy_scale = config.actor_entropy_scale # Used to compute actor_loss (???)
        self.grad_clip_norm = config.grad_clip # used during training (???)

        # self._model_initialize(config, state_dim, action_dim)
        # self._optim_initialize(config)

    def collect_seed_episodes(self, env):
        s, done  = env.reset(), False 
        for i in range(self.seed_steps):
            a = env.action_space.sample()
            ns, r, done, _ = env.step(a)
            if done:
                self.buffer.add(s,a,r,done)
                s, done  = env.reset(), False 
            else:
                self.buffer.add(s,a,r,done)
                s = ns    

    def act(self, state, prev_action, prev_rssm):
        state = state.reshape(1, 1, -1) # (1, 1, n_beams) # (single batch, single channel, n_beams)
        prev_action = torch.zeros(1, self.act_dim) if prev_action is None else torch.tensor(prev_action.reshape(1, -1))
        prev_rssmstate = self.RSSM._init_rssm_state(1) if prev_rssm is None else prev_rssm
        with torch.no_grad():
            embed = self.ObsEncoder(torch.FloatTensor(state)).unsqueeze(0) # Should be torch.float32
            _, posterior_rssm_state = self.RSSM.rssm_observe(embed.squeeze(0), prev_action, True, prev_rssmstate)
            model_state = self.RSSM.get_model_state(posterior_rssm_state)
            action, _ = self.ActionModel(model_state)
        
        return action

    def train(self):
        """ 
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        train_metrics = {}
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        if self.buffer.idx < self.batch_size * self.seq_len:
            return {}

        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to('cpu')                         #t, t+seq_len 
            actions = torch.tensor(actions, dtype=torch.float32).to('cpu')                 #t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to('cpu').unsqueeze(-1)   #t-1 to t+seq_len-1
            nonterms = torch.tensor(1-terms, dtype=torch.float32).to('cpu').unsqueeze(-1)  #t-1 to t+seq_len-1

            model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior = self.representation_loss(obs, actions, rewards, nonterms)
            
            self.model_optimizer.zero_grad()
            model_loss.backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
            self.model_optimizer.step()

            actor_loss, value_loss, target_info = self.actorcritc_loss(posterior)

            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_list), self.grad_clip_norm)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_list), self.grad_clip_norm)

            self.actor_optimizer.step()
            self.value_optimizer.step()

            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                post_ent = torch.mean(post_dist.entropy())

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info['mean_targ'])
            min_targ.append(target_info['min_targ'])
            max_targ.append(target_info['max_targ'])
            std_targ.append(target_info['std_targ'])

        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_loss']=np.mean(kl_l)
        train_metrics['reward_loss']=np.mean(reward_l)
        train_metrics['obs_loss']=np.mean(obs_l)
        train_metrics['value_loss']=np.mean(value_l)
        train_metrics['actor_loss']=np.mean(actor_l)
        train_metrics['prior_entropy']=np.mean(prior_ent_l)
        train_metrics['posterior_entropy']=np.mean(post_ent_l)
        train_metrics['pcont_loss']=np.mean(pcont_l)
        train_metrics['mean_targ']=np.mean(mean_targ)
        train_metrics['min_targ']=np.mean(min_targ)
        train_metrics['max_targ']=np.mean(max_targ)
        train_metrics['std_targ']=np.mean(std_targ)

        return train_metrics

    def actorcritc_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
        
        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        
        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        with FreezeParameters(self.world_list+self.value_list+[self.TargetValueModel]+[self.DiscountModel]):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_modelstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.discount*torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)

        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self._value_loss(imag_modelstates, discount, lambda_returns)     

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
        }

        return actor_loss, value_loss, target_info

    def representation_loss(self, obs, actions, rewards, nonterms):
        embed = self.ObsEncoder(obs)                                         #t to t+seq_len   
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)   
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len   
        obs_dist = self.ObsDecoder(post_modelstate[:-1])                     #t to t+seq_len-1  
        reward_dist = self.RewardDecoder(post_modelstate[:-1])               #t to t+seq_len-1  
        pcont_dist = self.DiscountModel(post_modelstate[:-1])                #t to t+seq_len-1   
        
        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = self.loss_scale['kl'] * div + reward_loss + obs_loss + self.loss_scale['discount']*pcont_loss
        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)
        
        if self.config.actor_grad == 'reinforce':
            advantage = (lambda_returns-imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1)) 
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates) 
        value_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss
            
    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save(self, directory="./saves"):
        save_dict = self.get_save_dict()
        filename = self.name
        save_path = os.path.join('%s/%s_model.pth' % (directory, filename))
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])
            
    def create_agent(self):
        obs_shape = (1, self.state_dim) # EMRAN I think it's input size without the window, and it'll use seq_len to fix, might not be though :/
        action_size = self.act_dim # EMRAN Same as this
        deter_size = self.config.rssm_info['deter_size']
        if self.config.rssm_type == 'continuous':
            stoch_size = self.config.rssm_info['stoch_size']
        elif self.config.rssm_type == 'discrete':
            category_size = self.config.rssm_info['category_size']
            class_size = self.config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = self.config.embedding_size
        rssm_node_size = self.config.rssm_node_size
        modelstate_size = stoch_size + deter_size 
    
        self.buffer = TransitionBuffer(self.config.capacity, obs_shape, action_size, self.seq_len, self.batch_size, self.config.obs_dtype, self.config.action_dtype)
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, 'cpu', self.config.rssm_type, self.config.rssm_info)
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, self.config.actor, self.config.expl)
        self.RewardDecoder = DenseModel((1,), modelstate_size, self.config.reward)
        self.ValueModel = DenseModel((1,), modelstate_size, self.config.critic)
        self.TargetValueModel = DenseModel((1,), modelstate_size, self.config.critic)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        
        if self.config.discount['use']:
            self.DiscountModel = DenseModel((1,), modelstate_size, self.config.discount)
        self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, self.config.obs_encoder)
        self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, self.config.obs_decoder)

        self._optim_initialize()

    def _optim_initialize(self):
        model_lr = self.config.lr['model']
        actor_lr = self.config.lr['actor']
        value_lr = self.config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr)

    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount['use']:
            print('\n Discount decoder: \n', self.DiscountModel)
        print('\n Actor: \n', self.ActionModel)
        print('\n Critic: \n', self.ValueModel)
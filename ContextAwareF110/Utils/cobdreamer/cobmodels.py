import copy
import torch
from torch import nn
import numpy as np

import time

import cobnetworks as cnetworks
import cobtools as tools

to_np = lambda x: x.detach().cpu().numpy()
CONTEXT_SIZE = 2
POLICY_FREQUENCY = 2
tau = 0.005

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions + CONTEXT_SIZE, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions + CONTEXT_SIZE, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, context):
        xu = torch.cat([state, action, context], -1)
        
        x1 = nn.functional.relu(self.linear1(xu))
        x1 = nn.functional.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = nn.functional.relu(self.linear4(xu))
        x2 = nn.functional.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    
    def to(self, device):
        # self.action_scale = self.action_scale.to(device)
        # self.action_bias = self.action_bias.to(device)
        # self.noise = self.noise.to(device)
        return super(QNetwork, self).to(device)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, max_action, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.max_action = max_action
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = nn.functional.relu(self.linear1(state))
        x = nn.functional.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class CtxMask(nn.Module):
    def __init__(self, obs_space, act_space, config, k=1, device='cuda'):
        super(CtxMask, self).__init__()   
        obs_space = obs_space[0][0]
        act_space = act_space[0][0] 
        h_size = 256
        window_in = 1
        window_out = 1
        lr = config.ctx_mask['lr'] # 0.0001

        self.alpha = 0.2
        self.gamma = 0.99
        self.it = 0
        self.automatic_entropy_tuning = True
        self.lr = config.ctx_mask['lr']

        self.critic = QNetwork(obs_space, act_space, h_size)
        self.critic_target = QNetwork(obs_space, act_space, h_size)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor((act_space, 1))).item() # EMRAN, may have to be (2,1) instead of (2,)
                self.log_alpha = torch.zeros(1, requires_grad=True)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.policy = GaussianPolicy(obs_space, act_space, 1, h_size)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
    
    def forward(self, state, noise=0.1):
        action, _, _ = self.policy.sample(state) # action, log_prob, mean
        action = nn.functional.sigmoid(action.data) #.numpy().flatten()
        # if noise != 0: 
        #     action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(0, 1)

    def _train(self, data):
        b, t, _ = data['image'].shape
        states = data['image'][:, :-1].reshape(-1, data['image'].shape[-1])
        next_states = data['image'][:, 1:].reshape(-1, data['image'].shape[-1])
        rewards = data['reward'][:, 1:].reshape(-1, 1)
        masks = data['mask'][:, :-1].reshape(-1, data['mask'].shape[-1])
        contexts = data['context'][:, :-1].reshape(-1, data['context'].shape[-1])
        next_contexts = data['context'][:, 1:].reshape(-1, data['context'].shape[-1])

        with torch.no_grad():
            # Select action according to policy and add clipped noise (for exploration) 
            next_action, next_state_log_pi, _ = self.policy.sample(next_states)
            next_action = next_action.clamp(0.0, 1.0)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_action, next_contexts)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_state_log_pi
            target_Q = rewards + (1.0 * self.gamma * target_Q) # made done = 1.0, pretty dure done was (1.0 - 0.0) in SAC # GAMMA = 0.99

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, masks, contexts)

        # Compute critic loss
        critic_loss = nn.functional.mse_loss(current_Q1, target_Q) + nn.functional.mse_loss(current_Q2, target_Q) 

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.policy.sample(states)

        qf1_pi, qf2_pi = self.critic(states, pi, contexts)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        pi, log_pi, _ = self.policy.sample(states)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach().cpu()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()


            self.alpha = self.log_alpha.exp() #.to(self.log_alpha.device)
            self.alpha = self.alpha.to(next_state_log_pi.device)
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        pi, log_pi, _ = self.policy.sample(states)

        # Every POLICY FREQUENCY, update critic weights
        if self.it % POLICY_FREQUENCY == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            pi, log_pi, _ = self.policy.sample(states)

        self.it += 1
        total_loss = policy_loss + critic_loss
        
        return total_loss
    

class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        # shapes =  obs_space # np.concatenate((obs_space, act_space), axis=1) # {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        shapes = {
            'is_first': (1,),
            'image': (108,),
            'action': (2,),
            'reward': (1,),
            'context': (2,),
            'is_terminal': (1,)
        }
        print("config.encoder :", config.encoder)
        self.encoder = cnetworks.MultiEncoder(shapes, **config.encoder)
        self.encoder._mlp.to(config.device)
        if hasattr(config, "add_dcontext") and config.add_dcontext: # EMRAN check this
            context_size = 2 # obs_space["context"].shape[0]
        self.embed_size = self.encoder.outdim
        self.dynamics = cnetworks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            act_space.flatten()[0], # config.num_actions,
            self.embed_size,
            config.device,
            add_dcontext=config.add_dcontext,
            add_dcontext_prior=config.add_dcontext_prior,
            add_dcontext_posterior=config.add_dcontext_posterior,
            context_size=context_size,
        )
        self.heads = nn.ModuleDict()
        print("config.decoder :", config.decoder)
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        if config.add_dcontext:
            feat_size += context_size
        self.heads["decoder"] = cnetworks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = cnetworks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = cnetworks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        # EMRAN 
        ## Use context head here in original implementaition
        if hasattr(config, "use_context_head") and config.use_context_head:
            assert config.add_dcontext
            self.head["context"] = nets.MLP(
                shapes["context"], **config.context_head, name="context"
            )
            assert "context" not in config.grad_heads
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

        # Context Mask
        self._ctx_mask = CtxMask(obs_space, act_space, config)
        # self._ctx_opt = tools.Optimizer(
        #     "_ctx_mask",
        #     self._ctx_mask.parameters(),
        #     config.ctx_mask["lr"],
        #     config.ctx_mask["eps"],
        #     config.ctx_mask["grad_clip"],
        #     config.weight_decay,
        #     opt=config.opt,
        #     use_amp=self._use_amp,
        # )

    def _train(self, data):
        # print("--------------------------------------------")
        # before = time.time()
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)
        dcontext = data['context']

        # print("self.preprocess(data) :", time.time() - before)
        # before = time.time()

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                # print("Encoder :", time.time() - before)
                # before = time.time()

                if self.dynamics._add_dcontext:
                    b, t, o = data['image'].shape
                    ctx_mask = self._ctx_mask(data['image'].reshape(-1, o)).reshape(b, t, -1)
                    dcontext *= ctx_mask

                    self._ctx_mask._train(data)


                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"], dcontext=dcontext if self.dynamics._add_dcontext else None
                )
                # print("Dynamics Observe :", time.time() - before)
                # before = time.time()
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                # print("Dynamics Loss :", time.time() - before)
                # before = time.time()
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    if name == "context":
                        print("\n\n\n\n\n\n ahhhhhh context head ahhhhh \n\n\n\n\n\n")
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    if not dcontext is None and self.dynamics._add_dcontext:
                        feat = torch.concatenate([feat, dcontext], -1)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                # print("Pred Loss:", time.time() - before)
                # before = time.time()
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                print(scaled)
                model_loss = sum(scaled.values()) + kl_loss
                # print("Scaled model loss :", time.time() - before)
                # before = time.time()
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        # print("Metrics update :", time.time() - before)
        # before = time.time()
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        # if hasattr(self.config, "use_context_head") and self.config.use_context_head:
        if "context" in self.heads:
            pure_context_head_fn = nj.pure(
                lambda ctx: self.heads['context'](ctx), nested=True
            )
            context_head_state = self.heads["context"].getm()
            adv_pred, _ = pure_context_head_fn(sg(context_head_state), nj.rng(), post)
            losses["context_adv"] = -adv_pred.log_prob(
                np.zeros_like(data['context'], np.float32)
            )
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        if type(obs) == np.ndarray:
            return torch.from_numpy(obs).float().to(self._config.device)
        # print("obs :", obs)
        # print("obs[context].shape :", obs["context"].shape)
        # print(obs)
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        # obs["image"] = #obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._add_dcontext = world_model.dynamics._add_dcontext
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        if self._add_dcontext:
            feat_size += world_model.dynamics.context_size

        self.actor = cnetworks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            device=config.device,
            name="Actor",
        )
        self.value = cnetworks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
        context
    ):
        self._update_slow_target()
        metrics = {}

        if self._add_dcontext:
            start['context'] = context # ['context']

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                # print("in ImagBehaviour._train -> self._config.actor[entropy], actor_ent :", self._config.actor['entropy'], actor_ent)
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items() if k in list(start.keys())}
        start = {k: torch.tensor(v, dtype=start['stoch'].dtype).to(start['stoch'].device) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            context = state['context'] if self._add_dcontext else None
            feat = dynamics.get_feat(state)
            if self._add_dcontext:
                feat = torch.concatenate([feat, state['context']], -1)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, dcontext=context)
            succ = {**succ, "context": context} if self._add_dcontext else succ
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            if self._world_model.dynamics._add_dcontext:
                inp = torch.concatenate([inp, imag_state['context']], -1)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
        dcontext=None,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

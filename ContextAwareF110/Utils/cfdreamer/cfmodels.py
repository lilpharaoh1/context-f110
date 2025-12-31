import copy
import torch
from torch import nn
import numpy as np

import time

import cfnetworks
import cftools as tools

to_np = lambda x: x.detach().cpu().numpy()

CONTEXT_SIZE = 64


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

class ForwardModel(nn.Module):
    def __init__(self, obs_space, act_space, config, device='cuda'):
        super(ForwardModel, self).__init__()
        self.layers = nn.Sequential()

        obs_size = obs_space[0][0]
        act_size = act_space[0][0]
        ctx_size = CONTEXT_SIZE
        layer_size = 256
        self.out_dim = obs_size
        self.epsilon = 1e-12
    
        # Block One
        self.layers.add_module(f"ctxenc_linear0", nn.Linear(obs_size + act_size + ctx_size, layer_size, bias=False).to(device))
        self.layers.add_module(f"ctxenc_norm0", nn.LayerNorm(layer_size, eps=1e-03).to(device))
        self.layers.add_module(f"ctxenc_act0", nn.SiLU())
        
        # Block Two
        self.layers.add_module(f"ctxenc_linear1", nn.Linear(layer_size, layer_size, bias=False).to(device))
        self.layers.add_module(f"ctxenc_norm1", nn.LayerNorm(layer_size, eps=1e-03).to(device))
        self.layers.add_module(f"ctxenc_act1", nn.SiLU())

        # Out Block
        self.layers.add_module(f"ctxenc_linear_out", nn.Linear(layer_size, 2 * obs_size, bias=False).to(device))
        self.layers.add_module(f"ctxenc_act_out", nn.SiLU())


    def forward(self, obs, action, context):
        x = torch.concatenate([obs, action, context], -1) 
        x = self.layers(x) + self.epsilon

        return {"mean": x[:, :self.out_dim], "var": x[:, self.out_dim:]}

class ContextEncoder(nn.Module):
    def __init__(self, obs_space, config, k=1, device='cuda'):
        super(ContextEncoder, self).__init__()
        self.layers = nn.Sequential()
    
        obs_size = obs_space[0][0]
        ctx_size = CONTEXT_SIZE
        layer_size = 256
        self.k=k

        # Block One
        self.layers.add_module(f"ctxenc_linear0", nn.Linear(k * obs_size, layer_size, bias=False).to(device))
        self.layers.add_module(f"ctxenc_norm0", nn.LayerNorm(layer_size, eps=1e-03).to(device))
        self.layers.add_module(f"ctxenc_act0", nn.SiLU())
        
        # Block Two
        self.layers.add_module(f"ctxenc_linear1", nn.Linear(layer_size, layer_size, bias=False).to(device))
        self.layers.add_module(f"ctxenc_norm1", nn.LayerNorm(layer_size, eps=1e-03).to(device))
        self.layers.add_module(f"ctxenc_act1", nn.SiLU())

        # Out Block
        self.layers.add_module(f"ctxenc_linear_out", nn.Linear(layer_size, ctx_size, bias=False).to(device))
        self.layers.add_module(f"ctxenc_act_out", nn.SiLU())


    def forward(self, obs):
        return self.layers(obs)


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
            'context': (CONTEXT_SIZE,),
            'is_terminal': (1,)
        }
        print("config.encoder :", config.encoder)
        self.encoder = cfnetworks.MultiEncoder(shapes, **config.encoder)
        self.encoder._mlp.to(config.device)
        if hasattr(config, "add_dcontext") and config.add_dcontext: # EMRAN check this
            context_size = CONTEXT_SIZE # obs_space["context"].shape[0]
        self.embed_size = self.encoder.outdim
        self.dynamics = cfnetworks.RSSM(
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
        self.heads["decoder"] = cfnetworks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = cfnetworks.MLP(
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
        self.heads["cont"] = cfnetworks.MLP(
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

        self._backward_model = ForwardModel(obs_space, act_space, config)
        self._forward_model = ForwardModel(obs_space, act_space, config)
        self._ctx_encoder = ContextEncoder(obs_space, config)
        
        self._backward_opt = tools.Optimizer(
            "_backward_model",
            self._backward_model.parameters(),
            config.ctx_encoder["lr"],
            config.ctx_encoder["eps"],
            config.ctx_encoder["grad_clip"],
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._forward_opt = tools.Optimizer(
            "_forward_model",
            self._forward_model.parameters(),
            config.ctx_encoder["lr"],
            config.ctx_encoder["eps"],
            config.ctx_encoder["grad_clip"],
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._ctx_opt = tools.Optimizer(
            "_ctx_encoder",
            self._ctx_encoder.parameters(),
            config.ctx_encoder["lr"],
            config.ctx_encoder["eps"],
            config.ctx_encoder["grad_clip"],
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

    def _train(self, data):
        # print("--------------------------------------------")
        # before = time.time()
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)
        # dcontext = self._ctx_encoder(data['image'].reshape(-1, data['image'].shape[-1]))\
            # .reshape((data['image'].shape[0], data['image'].shape[1], -1))
        # data['context'] = dcontext
        dcontext = data['context']


        # print("self.preprocess(data) :", time.time() - before)
        # before = time.time()

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                # print("Encoder :", time.time() - before)
                # before = time.time()
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
                if self.dynamics._add_dcontext:
                    torch.autograd.set_detect_anomaly(True)
                    b, l, o = data['image'].shape
                    preds['forward'] = self._forward_model(data['image'][:, :-1].reshape(-1, o), data['action'][:, :-1].reshape(-1, 2), data['context'][:, :-1].reshape(-1, CONTEXT_SIZE))
                    preds['backward'] = self._backward_model(data['image'][:, 1:].reshape(-1, o), data['action'][:, :-1].reshape(-1, 2), data['context'][:, :-1].reshape(-1, CONTEXT_SIZE))
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
                if self.dynamics._add_dcontext:
                    b, l, o = data['image'].shape
                    states, next_states = data['image'][:, :-1].reshape(-1, o), data['image'][:, 1:].reshape(-1, o)
                    def nll(target, pred):
                        return -0.5 * (
                            ((target - pred['mean']) ** 2 / torch.exp(0.5 * pred['var'])).sum(dim=1)
                            + pred['var'].sum(dim=1)
                            + pred['mean'].shape[-1] * torch.log(torch.tensor(2 * torch.pi))
                        )
                        
                    # print("WorldModel._train -> preds[forward], next_states:", preds['forward'].shape, next_states.shape)
                    # print("WorldModel._train -> preds[backward], sstates:", preds['backward'].shape, states.shape)
                    forward_loss = nll(next_states, preds['forward'])
                    backward_loss = nll(states, preds['backward'])
                    ctx_loss = torch.mean(forward_loss) + torch.mean(0.5 * backward_loss)
                for name, pred in preds.items():
                    if name in ['forward', 'backward']:
                        continue
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                # print("Pred Loss:", time.time() - before)
                # before = time.time()
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                # print("Scaled model loss :", time.time() - before)
                # before = time.time()
            _ = self._ctx_opt(ctx_loss, self._ctx_encoder.parameters())
            _ = self._forward_opt(torch.mean(forward_loss), self._forward_model.parameters())
            _ = self._backward_opt(torch.mean(backward_loss), self._backward_model.parameters())
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
        metrics["ctx_loss"] = to_np(ctx_loss)
        metrics["forward_loss"] = to_np(forward_loss)
        metrics["backward_loss"] = to_np(backward_loss)
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

        self.actor = cfnetworks.MLP(
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
        self.value = cfnetworks.MLP(
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

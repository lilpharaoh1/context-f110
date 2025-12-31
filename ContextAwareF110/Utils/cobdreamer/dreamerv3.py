import argparse
import functools
import os
import pathlib
import sys
import collections
import glob

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import yaml # import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import cobmodels as cmodels
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

to_np = lambda x: x.detach().cpu().numpy()

# class DreamBuffer:
    # def __init__(self):



class DreamerV3(nn.Module):
    def __init__(self, obs_space, act_space, name, max_action=1, window_in=1, window_out=1, multiagent=True, lr=None):
        super(DreamerV3, self).__init__()
        torch.use_deterministic_algorithms(False)
        self.name = name

        full_path = "ContextAwareF110/Utils/cobdreamer/configs.yaml"  
        with open(full_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader) # Load config

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

            return base

        name_list = ["defaults", "multi-agent"] if multiagent else ["defaults", "single-agent"]
        defaults = {}
        for name in name_list:
            defaults = recursive_update(defaults, config[name])

        config = argparse.Namespace(**defaults)
        config.num_actions = act_space
        config.model_lr = lr if not lr is None else config.model_lr
        self._config = config

        obs_space, act_space = np.array(obs_space).reshape(1, -1), np.array(act_space).reshape(1, -1)

        self.buffer_eps = collections.OrderedDict()
        self.buffer_epnum = 0
        self.buffer_ptr = 0
        # self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._expl_coeff = 0.99
        self._metrics = {}
        # this is update step
        self._step = 0 # logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = None
        self._wm = cmodels.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = cmodels.ImagBehavior(config, self._wm).to(self._config.device)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        # Context Mask
        # self._ctx_mask = cmodels.CtxMask(obs_space, config)

    def act(self, obs, action, latent, context=None, is_first=False, video=False):
        obs = {
            "is_first": np.array([1.0 if is_first else 0.0]),
            "image" : obs.reshape(1, -1),
            "action": action.reshape(1, -1) if not action is None else np.zeros((1, 2)) ,
            "context": context.reshape(1, -1) if not context is None else np.zeros((1, 2)),
            "is_terminal": np.array([0.0])
        }

        obs, action, latent = self._wm.preprocess(obs), action.to(self._config.device) if not action is None else action, latent # .to(self._config.device) if not latent is None else latent
        embed = self._wm.encoder(obs)
        dcontext = None
        if self._wm.dynamics._add_dcontext:
            ctx_mask = self._wm._ctx_mask(obs['image'])
            dcontext = obs['context'] * ctx_mask

            # Reshape for context buffer
            ctx_mask = ctx_mask.cpu().reshape(1, -1)

        # dcontext = obs["context"] if self._wm.dynamics._add_dcontext else None
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"], dcontext=dcontext)
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not dcontext is None and self._wm.dynamics._add_dcontext:
            feat = torch.concatenate([feat, dcontext], -1)

        if video:
            # states, _ = self._wm.dynamics.observe(
            #     embed[:6, :5], action, data["is_first"][:6, :5]
            # )
            recon = self._wm.heads["decoder"](feat)["image"].mode().cpu().detach().numpy().reshape(-1)
        
        # if self._should_expl(self._step):
        # if np.random.uniform(0, 1) < self._expl_coeff:
        if False:
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        self._step += 1
        
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        if video:
            return action, latent, recon
        return action, latent, ctx_mask
        # return policy_output, state
 

    def __call__(self, obs, action, latent, context=None, is_first=False):
        return self.act(obs, action, latent, context=context, is_first=is_first)

    # def __call__(self, obs, reset, state=None, training=True):
    #     step = self._step
    #     if training:
    #         steps = (
    #             self._config.pretrain
    #             if self._should_pretrain()
    #             else self._should_train(step)
    #         )
    #         for _ in range(steps):
    #             self._train(next(self._dataset))
    #             self._update_count += 1
    #             self._metrics["update_count"] = self._update_count
    #         if self._should_log(step):
    #             for name, values in self._metrics.items():
    #                 self._logger.scalar(name, float(np.mean(values)))
    #                 self._metrics[name] = []
    #             if self._config.video_pred_log:
    #                 openl = self._wm.video_pred(next(self._dataset))
    #                 self._logger.video("train_openl", to_np(openl))
    #             self._logger.write(fps=True)

    #     policy_output, state = self._policy(obs, state, training)

    #     if training:
    #         self._step += len(reset)
    #         self._logger.step = self._config.action_repeat * self._step
    #     return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            self._step += 1
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def train(self):
        if len(self.buffer_eps.keys()) < 1: #self._config.batch_size * 2:
            return
        metrics = {}
        self._dataset = make_dataset(self.buffer_eps, self._config)
        data = next(self._dataset)
        # if self._wm.dynamics._add_dcontext:
        #     b, t, o = data['image'].shape
        #     ctx_mask = self._ctx_mask(torch.tensor(data['image'].reshape(-1, o), dtype=torch.float32).to(self._config.device))
        #     ctx_mask[ctx_mask >= 0.5], ctx_mask[ctx_mask<0.5] = 1.0, 0.0
        #     data["context"] *= ctx_mask
        # unsqueeze is_frist, is_terminal and reward EMRAN hack fix again
        for k, v in data.items():
            if v.shape[-1] == 1:
                data[k] = v.squeeze(-1)
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        # reward = lambda f, s, a: self._wm.heads["reward"](
            # self._wm.dynamics.get_feat(s)
        # ).mode()
        def reward(f, s, a):
            feat = self._wm.dynamics.get_feat(s)
            if self._wm.dynamics._add_dcontext:
                feat = torch.concatenate([feat, s['context']], -1)
            return self._wm.heads["reward"](feat).mode()
        metrics.update(self._task_behavior._train(start, reward, data['context'])[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def save(self, path):
        eps_dir = path + "/Buffer"
        eps_name = 'eps_' + str(self.buffer_ptr)
        if not os.path.exists(eps_dir):
            os.mkdir(eps_dir)
        tools.save_episodes(eps_dir, {eps_name: self.buffer_eps[eps_name]})
        self.buffer_ptr += 1 

        items_to_save = {
            "agent_state_dict": self.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(self),
        }
        torch.save(items_to_save, path + "/" + self.name + ".pth")
        
        # Change expl_coeff every episode
        self._expl_coeff *= 0.99

    def load(self, path):
        eps_dir = path + "/Buffer/*"
        eps_paths = glob.glob(eps_dir)
        laps = [int(eps_path.split('/')[-1][4:].split('-')[0]) for eps_path in eps_paths]
        eps_paths = [s for _, s in sorted(zip(laps, eps_paths))]

        for idx, eps_path in enumerate(eps_paths):
            with np.load(eps_path) as data:
                episode = {k: data[k] for k in data.files}
            self.buffer_eps[f"eps_{idx}"] = episode

        checkpoint = torch.load(path + '/' + self.name + ".pth")
        self.load_state_dict(checkpoint['agent_state_dict'])

        # EMRAN need something better than this
        self._expl_coeff = 0




def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


# def main(config):
#     tools.set_seed_everywhere(config.seed)
#     if config.deterministic_run:
#         tools.enable_deterministic_run()
#     logdir = pathlib.Path(config.logdir).expanduser()
#     config.traindir = config.traindir or logdir / "train_eps"
#     config.evaldir = config.evaldir or logdir / "eval_eps"
#     config.steps //= config.action_repeat
#     config.eval_every //= config.action_repeat
#     config.log_every //= config.action_repeat
#     config.time_limit //= config.action_repeat

#     print("Logdir", logdir)
#     logdir.mkdir(parents=True, exist_ok=True)
#     config.traindir.mkdir(parents=True, exist_ok=True)
#     config.evaldir.mkdir(parents=True, exist_ok=True)
#     step = count_steps(config.traindir)
#     # step in logger is environmental step
#     logger = tools.Logger(logdir, config.action_repeat * step)

#     print("Create envs.")
#     if config.offline_traindir:
#         directory = config.offline_traindir.format(**vars(config))
#     else:
#         directory = config.traindir
#     train_eps = tools.load_episodes(directory, limit=config.dataset_size)
#     if config.offline_evaldir:
#         directory = config.offline_evaldir.format(**vars(config))
#     else:
#         directory = config.evaldir
#     eval_eps = tools.load_episodes(directory, limit=1)
#     make = lambda mode, id: make_env(config, mode, id)
#     train_envs = [make("train", i) for i in range(config.envs)]
#     eval_envs = [make("eval", i) for i in range(config.envs)]
#     if config.parallel:
#         train_envs = [Parallel(env, "process") for env in train_envs]
#         eval_envs = [Parallel(env, "process") for env in eval_envs]
#     else:
#         train_envs = [Damy(env) for env in train_envs]
#         eval_envs = [Damy(env) for env in eval_envs]
#     acts = train_envs[0].action_space
#     print("Action Space", acts)
#     config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

#     state = None
#     if not config.offline_traindir:
#         prefill = max(0, config.prefill - count_steps(config.traindir))
#         print(f"Prefill dataset ({prefill} steps).")
#         if hasattr(acts, "discrete"):
#             random_actor = tools.OneHotDist(
#                 torch.zeros(config.num_actions).repeat(config.envs, 1)
#             )
#         else:
#             random_actor = torchd.independent.Independent(
#                 torchd.uniform.Uniform(
#                     torch.tensor(acts.low).repeat(config.envs, 1),
#                     torch.tensor(acts.high).repeat(config.envs, 1),
#                 ),
#                 1,
#             )

#         def random_agent(o, d, s):
#             action = random_actor.sample()
#             logprob = random_actor.log_prob(action)
#             return {"action": action, "logprob": logprob}, None

#         state = tools.simulate(
#             random_agent,
#             train_envs,
#             train_eps,
#             config.traindir,
#             logger,
#             limit=config.dataset_size,
#             steps=prefill,
#         )
#         logger.step += prefill * config.action_repeat
#         print(f"Logger: ({logger.step} steps).")

#     print("Simulate agent.")
#     train_dataset = make_dataset(train_eps, config)
#     eval_dataset = make_dataset(eval_eps, config)
#     agent = Dreamer(
#         train_envs[0].observation_space,
#         train_envs[0].action_space,
#         config,
#         logger,
#         train_dataset,
#     ).to(config.device)
#     agent.requires_grad_(requires_grad=False)
#     if (logdir / "latest.pt").exists():
#         checkpoint = torch.load(logdir / "latest.pt")
#         agent.load_state_dict(checkpoint["agent_state_dict"])
#         tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
#         agent._should_pretrain._once = False

#     # make sure eval will be executed once after config.steps
#     while agent._step < config.steps + config.eval_every:
#         logger.write()
#         if config.eval_episode_num > 0:
#             print("Start evaluation.")
#             eval_policy = functools.partial(agent, training=False)
#             tools.simulate(
#                 eval_policy,
#                 eval_envs,
#                 eval_eps,
#                 config.evaldir,
#                 logger,
#                 is_eval=True,
#                 episodes=config.eval_episode_num,
#             )
#             if config.video_pred_log:
#                 video_pred = agent._wm.video_pred(next(eval_dataset))
#                 logger.video("eval_openl", to_np(video_pred))
#         print("Start training.")
#         state = tools.simulate(
#             agent,
#             train_envs,
#             train_eps,
#             config.traindir,
#             logger,
#             limit=config.dataset_size,
#             steps=config.eval_every,
#             state=state,
#         )
#         items_to_save = {
#             "agent_state_dict": agent.state_dict(),
#             "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
#         }
#         torch.save(items_to_save, logdir / "latest.pt")
#     for env in train_envs + eval_envs:
#         try:
#             env.close()
#         except Exception:
#             pass


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--configs", nargs="+")
#     args, remaining = parser.parse_known_args()
#     configs = yaml.safe_load(
#         (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
#     )

#     def recursive_update(base, update):
#         for key, value in update.items():
#             if isinstance(value, dict) and key in base:
#                 recursive_update(base[key], value)
#             else:
#                 base[key] = value

#     name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
#     defaults = {}
#     for name in name_list:
#         recursive_update(defaults, configs[name])
#     parser = argparse.ArgumentParser()
#     for key, value in sorted(defaults.items(), key=lambda x: x[0]):
#         arg_type = tools.args_type(value)
#         parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
#     main(parser.parse_args(remaining))

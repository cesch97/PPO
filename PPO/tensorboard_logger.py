"""
TensorBoard logging shared by PPO and SelfPlayPPO. It is an addition to the notebook plots, not a replacement.

Everything is written against the number of frames (half-moves for the chess agent), so a run that is resumed with the
same log directory continues its curves. View it with `tensorboard --logdir logs/tb` or, in a notebook,
`%load_ext tensorboard` and `%tensorboard --logdir logs/tb`.

What is logged:
 - SCALARS: train/ (losses, kl, clip fraction, entropy, grad norm, lr, ...), rollout/ (episode return and length,
   explained variance, ...), eval/, time/, norms/ (global and per-module weight and gradient norms) and the others
   added by the trainers. A CUSTOM SCALARS tab groups the most useful curves.
 - HISTOGRAMS / DISTRIBUTIONS: weights/ and grads/ of every parameter, plus rollout histograms (actions, values).
 - TEXT: hyper-parameters, run info (device, versions, ...), the network architecture and a table of its parameters.
 - GRAPH: the network graph (traced with a sample observation).
 - IMAGES: a sample observation (one tile per channel / frame / plane).
 - HPARAMS: the hyper-parameters of the run, to compare runs against their final metrics.
"""
import math
import platform
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def explained_variance(values, returns):
    """1 means the critic predicts the returns perfectly, 0 is as good as a constant, < 0 is worse."""
    variance = returns.var(correction=0)
    if variance < 1e-8:
        return float('nan')
    return float(1 - (returns - values).var(correction=0) / variance)


def _md_cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines += ['| ' + ' | '.join(_md_cell(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def _hparam_value(value):
    """Values accepted by the HParams plugin (others are skipped)."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (tuple, list, np.ndarray)):
        return str(value)
    return None


class _Traced(nn.Module):
    """Gives add_graph a module that returns plain tensors (the networks' forward returns a distribution)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.raw_outputs(x)


class TensorBoardLogger:

    def __init__(self, log_dir, histograms=True, flush_secs=30):
        """histograms=False skips the weight and gradient histograms (the most expensive part for big networks)."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError('TensorBoard logging needs the tensorboard package: pip install tensorboard') from error
        self.log_dir = str(log_dir)
        self.histograms = histograms
        self.writer = SummaryWriter(self.log_dir, flush_secs=flush_secs)

    @staticmethod
    def resolve(tensorboard, **kwargs):
        """None disables logging, a path creates a logger, a TensorBoardLogger is used as it is."""
        if tensorboard is None or isinstance(tensorboard, TensorBoardLogger):
            return tensorboard
        return TensorBoardLogger(Path(tensorboard), **kwargs)

    # ------------------------------------------------------------------------------------------ setup (once)
    def log_setup(self, model, example_input, hparams=None, metrics=(), run_info=None, layout=None):
        """
        model: a network with a `raw_outputs(x)` method returning tensors, example_input: a batch of one observation.
        metrics: tags of the scalars that the HParams tab shows next to the hyper-parameters.
        layout: custom scalars layout, {category: {chart: [tag or regex, ...]}}.
        """
        hparams = {k: v for k, v in (hparams or {}).items() if _hparam_value(v) is not None}
        info = {'log dir': self.log_dir, 'device': next(model.parameters()).device, 'torch': torch.__version__,
                'python': platform.python_version(), **(run_info or {})}
        self.writer.add_text('setup/hyperparameters',
                             _md_table(['name', 'value'], [(k, v) for k, v in hparams.items()]), 0)
        self.writer.add_text('setup/run', _md_table(['name', 'value'], info.items()), 0)
        self._log_architecture(model)
        self._log_graph(model, example_input)
        if layout:
            self.writer.add_custom_scalars(layout)
        if hparams or metrics:
            from torch.utils.tensorboard.summary import hparams as hparams_summary
            summaries = hparams_summary({k: _hparam_value(v) for k, v in hparams.items()}, {m: 0 for m in metrics})
            for summary in summaries:
                self.writer._get_file_writer().add_summary(summary)
        self.writer.flush()

    def _log_architecture(self, model):
        params = list(model.named_parameters())
        total = sum(p.numel() for _, p in params)
        trainable = sum(p.numel() for _, p in params if p.requires_grad)
        buffers = sum(b.numel() for b in model.buffers())
        size_mb = sum(p.numel() * p.element_size() for _, p in params) / 2 ** 20

        modules = {}
        for name, param in params:
            modules.setdefault(name.split('.')[0], []).append(param)
        children = dict(model.named_children())
        summary = [(name, type(children[name]).__name__ if name in children else 'Parameter',
                    sum(p.numel() for p in ps), '%.1f%%' % (100 * sum(p.numel() for p in ps) / total))
                   for name, ps in modules.items()]
        summary.append(('**total**', type(model).__name__, '**%s**' % format(total, ','), '100%'))

        self.writer.add_text('model/architecture', '```\n%s\n```' % model, 0)
        self.writer.add_text('model/summary',
                             '%s trainable parameters of %s (%.2f MB, %s non-parameter buffer values)\n\n%s' % (
                                 format(trainable, ','), format(total, ','), size_mb, format(buffers, ','),
                                 _md_table(['module', 'type', 'parameters', 'share'], summary)), 0)
        self.writer.add_text('model/parameters', _md_table(
            ['parameter', 'shape', 'numel', 'trainable'],
            [(name, tuple(p.shape), p.numel(), p.requires_grad) for name, p in params]), 0)
        self.writer.add_scalar('model/num_parameters', total, 0)

    def _log_graph(self, model, example_input):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # data dependent branches are traced with the example input
                self.writer.add_graph(_Traced(model), example_input, use_strict_trace=False)
        except Exception as error:
            warnings.warn('the network graph could not be traced, it is not logged: %r' % error)

    def log_observation(self, obs, tag='setup/observation', min_size=64):
        """A tile per channel of an observation with shape (C, H, W): stacked frames, board planes, ..."""
        tiles = np.asarray(obs, dtype=np.float32)
        if tiles.ndim != 3:
            return
        tiles = tiles - tiles.min()
        if tiles.max() > 0:
            tiles = tiles / tiles.max()
        scale = max(1, math.ceil(min_size / max(tiles.shape[1:])))
        tiles = tiles.repeat(scale, axis=1).repeat(scale, axis=2)
        self.writer.add_images(tag, tiles[:, None], 0, dataformats='NCHW')

    # ------------------------------------------------------------------------------------------ during training
    def scalars(self, step, values, prefix=''):
        """Logs the finite numbers of a dict."""
        for name, value in values.items():
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                self.writer.add_scalar(prefix + name, value, step)

    def update(self, step, opt_info, log_std=False):
        """The dict returned by ppo_update. log_std: also log the std of every action (continuous actions only)."""
        self.scalars(step, {k: v for k, v in opt_info.items() if k != 'std'}, 'train/')
        if log_std:
            std = opt_info['std'].flatten() * opt_info.get('std_coef', 1.0)
            self.scalars(step, {'action_%i' % i: s for i, s in enumerate(std.tolist())}, 'policy/std/')

    def histogram(self, step, tag, values, bins='tensorflow'):
        values = torch.as_tensor(values).detach().float().flatten()
        if values.numel() and torch.isfinite(values).all():
            self.writer.add_histogram(tag, values, step, bins=bins)

    def weights(self, step, model):
        """Histograms of weights and gradients (the gradients are the ones of the last minibatch of the update)."""
        norms = {'weights': [], 'grads': []}
        module_grads = {}
        for name, param in model.named_parameters():
            tag = name.replace('.', '/')
            norms['weights'].append(param.detach().float().norm())
            if self.histograms:
                self.histogram(step, 'weights/' + tag, param)
            if param.grad is not None:
                norms['grads'].append(param.grad.float().norm())
                module_grads.setdefault(name.split('.')[0], []).append(norms['grads'][-1])
                if self.histograms:
                    self.histogram(step, 'grads/' + tag, param.grad)
        for kind, values in norms.items():
            if values:
                self.scalars(step, {'global_' + kind: torch.stack(values).norm().item()}, 'norms/')
        for module, values in module_grads.items():
            self.scalars(step, {module: torch.stack(values).norm().item()}, 'norms/grad_')

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

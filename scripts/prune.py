"""Structured channel pruning of a trained VoxHOD model to a smaller VOX_WIDTH.

Builds the VOX_WIDTH model that train.py would build and fills it with the most
important channels of a trained wider model, so train.py can fine-tune it
(VOX_WARM_START=<pruned file>). The target is exactly the VOX_WIDTH architecture, so a
model trained from scratch at the same width with the same budget is the fair control.

Channel importance (Network Slimming): |gamma| of the BatchNorms. Layers whose outputs
meet in an Add (residual blocks, the feature-pyramid sum) must keep the same channels,
so the graph is split into "channel streams": a Conv2D starts a new stream, BatchNorm,
ReLU, DepthwiseConv2D and UpSampling2D pass their input stream on, and Add merges the
streams of its inputs. Each stream keeps the channels with the highest sum of
mean-normalised |gamma| over all BatchNorms in it. The input (3) and the two heads
(heat 1, size 2) are never pruned.

    VOX_WIDTH=0.5 python prune.py ../artifacts_v3/voxhod.keras pruned_05.keras
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import train  # noqa: E402  (reads VOX_WIDTH at import)

L = tf.keras.layers
PASS = (L.BatchNormalization, L.ReLU, L.DepthwiseConv2D, L.UpSampling2D)


def streams(model: tf.keras.Model):
    """Per layer: (input stream id, output stream id), with Add-merged streams unified."""
    parent: dict[int, int] = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tensor_stream, io, next_id = {}, {}, 0
    for layer in model.layers:
        ins = layer.input if isinstance(layer.input, list) else [layer.input]
        in_ids = [tensor_stream.get(t.name) for t in ins] if not isinstance(layer, L.InputLayer) else []
        if isinstance(layer, L.InputLayer) or isinstance(layer, L.Conv2D) and not isinstance(layer, L.DepthwiseConv2D):
            out = next_id; parent[out] = out; next_id += 1
        elif isinstance(layer, PASS):
            out = in_ids[0]
        elif isinstance(layer, L.Add):
            out = in_ids[0]
            for other in in_ids[1:]:
                parent[find(other)] = find(out)
        elif isinstance(layer, L.Concatenate):
            out = None
        else:
            raise ValueError(f"unhandled layer {layer.name} ({type(layer).__name__})")
        tensor_stream[layer.output.name] = out
        io[layer.name] = (in_ids[0] if in_ids else None, out)
    resolve = lambda s: None if s is None else find(s)
    return [(layer, resolve(io[layer.name][0]), resolve(io[layer.name][1])) for layer in model.layers]


def main() -> None:
    source = tf.keras.models.load_model(sys.argv[1], compile=False)
    target = train.build_model()
    s_info, t_info = streams(source), streams(target)
    assert [type(a[0]) for a in s_info] == [type(b[0]) for b in t_info], "architectures differ"

    width, score = {}, {}
    for (s_layer, _, s_out), (t_layer, _, _) in zip(s_info, t_info):
        if isinstance(t_layer, L.Conv2D) and not isinstance(t_layer, L.DepthwiseConv2D):
            width[s_out] = t_layer.filters
        if isinstance(s_layer, L.BatchNormalization):
            g = np.abs(s_layer.get_weights()[0])
            score[s_out] = score.get(s_out, 0) + g / g.mean()
    input_stream = s_info[0][2]
    keep = {input_stream: np.arange(3)}
    for s, n in width.items():
        if s == input_stream:
            continue
        if s not in score or n == len(score[s]):  # heads: no BatchNorm, never pruned
            keep[s] = np.arange(n)
        else:
            keep[s] = np.sort(np.argsort(-score[s])[:n])
            print(f"stream {s}: {len(score[s])} -> {n} channels, kept score share {score[s][keep[s]].sum() / score[s].sum():.1%}")

    for (s_layer, s_in, s_out), (t_layer, _, _) in zip(s_info, t_info):
        weights = s_layer.get_weights()
        if not weights:
            continue
        if isinstance(s_layer, L.DepthwiseConv2D):
            new = [weights[0][:, :, keep[s_in], :]] + [w[keep[s_in]] for w in weights[1:]]
        elif isinstance(s_layer, L.Conv2D):
            new = [weights[0][:, :, keep[s_in], :][..., keep[s_out]]] + [w[keep[s_out]] for w in weights[1:]]
        elif isinstance(s_layer, L.BatchNormalization):
            new = [w[keep[s_out]] for w in weights]
        else:
            raise ValueError(f"weights in unhandled layer {s_layer.name}")
        t_layer.set_weights(new)

    print(f"parameters: {source.count_params():,} -> {target.count_params():,}")
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    target.save(sys.argv[2])
    print(f"Wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()

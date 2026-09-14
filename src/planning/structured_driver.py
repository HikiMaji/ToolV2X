"""Shared numeric cooperative planner built only from causal numeric inputs."""
import copy
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from planning.driver_contract import validate_numeric_output
from planning.structured_inputs import (StructuredDriverSpec,
                                        build_structured_plan_input,
                                        validate_structured_prepared)
from tools.task_spec import _check_json_native, _version_pair


DEFAULT_MODEL_VERSION = dict(name='structured_planner_network', revision='v1',
    training=dict(status='initialized_untrained', optimizer_steps=0))
_BATCH_KEYS = ('scene_patches', 'scene_mask', 'observations', 'observation_mask',
    'observation_sources', 'forecasts', 'forecast_mask', 'forecast_context',
    'entity_mask', 'ego_motion', 'ego_history', 'ego_history_mask',
    'previous_plan', 'previous_plan_valid')


def _numeric_tensor(value, shape, device, name):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            raise ValueError(name + ' must be numeric')
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError('invalid ' + name)
        tensor = torch.as_tensor(array, device=device, dtype=torch.float32)
    if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all().item():
        raise ValueError('invalid ' + name)
    return tensor


def _bool_tensor(value, shape, device, name):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.bool:
            raise ValueError(name + ' must be boolean')
        tensor = value.to(device=device)
    else:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype(bool):
            raise ValueError('invalid ' + name)
        tensor = torch.as_tensor(array, device=device, dtype=torch.bool)
    if tuple(tensor.shape) != shape:
        raise ValueError('invalid ' + name)
    return tensor


def _feature_history(features, prepared):
    names = ('ego_pose_history', 'ego_pose_history_valid', 'ego_pose_history_times')
    present = [name in features for name in names]
    if any(present) and not all(present):
        raise ValueError('ego pose history arrays must be supplied together')
    expected = prepared['ego_history_used']
    if not any(present):
        if expected is not None:
            raise ValueError('prepared ego history is absent from feature inputs')
        return
    arrays = []
    for name in names:
        value = features[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays.append(np.asarray(value))
    states, valid, times = arrays
    if (states.shape != (11, 3) or states.dtype.kind not in 'fiu' or
            not np.isfinite(states).all() or valid.shape != (11,) or
            valid.dtype != np.dtype(bool) or times.shape != (11,) or
            times.dtype.kind not in 'fiu' or not np.isfinite(times).all()):
        raise ValueError('invalid ego pose history feature layout')
    if expected is None or (not np.allclose(states, np.asarray(expected['states'])) or
            valid.tolist() != expected['valid'] or
            not np.allclose(times, np.asarray(expected['times']))):
        raise ValueError('feature and prepared ego histories differ')


def collate_structured_inputs(features_list, prepared_list, device='cpu'):
    """Collate validated samples and reproduce the original 5x4 shallow patches."""
    if (not isinstance(features_list, (list, tuple)) or
            not isinstance(prepared_list, (list, tuple)) or
            not features_list or len(features_list) != len(prepared_list)):
        raise ValueError('matching nonempty feature/prepared samples required')
    device = torch.device(device)
    spec_dict = prepared_list[0].get('driver_spec') if isinstance(prepared_list[0], dict) else None
    spec = StructuredDriverSpec.from_dict(spec_dict)
    tensors = {key: [] for key in ('observations', 'observation_mask',
        'observation_sources', 'forecasts', 'forecast_mask', 'forecast_context',
        'entity_mask')}
    scene_patches, scene_masks = [], []
    motions, histories, history_masks, prior_plans, prior_valid = [], [], [], [], []
    e, c = spec.max_entities, spec.max_forecast_sets_per_entity
    for features, prepared in zip(features_list, prepared_list):
        validate_structured_prepared(prepared)
        if prepared['driver_spec'] != spec_dict or not isinstance(features, dict):
            raise ValueError('structured batch specifications differ')
        regression = _numeric_tensor(features.get('regression_map'),
            (1, 2, 1, 14, 50, 88), device, 'regression_map')
        classification = _numeric_tensor(features.get('classification_map'),
            (1, 2, 1, 2, 50, 88), device, 'classification_map')
        active = _bool_tensor(features.get('active_agent_mask'), (1, 2, 1),
                              device, 'active_agent_mask')
        if not active[0, 0, 0].item():
            raise ValueError('current ego feature frame must be active')
        merged = torch.cat([regression, classification], dim=3).reshape(2, 16, 50, 88)
        patches = F.unfold(merged, kernel_size=(5, 4), stride=(5, 4)).transpose(1, 2)
        mask = active.reshape(2, 1).expand(2, 220)
        scene_patches.append(torch.where(mask.unsqueeze(-1), patches, torch.zeros_like(patches)))
        scene_masks.append(mask)

        raw = prepared['tensor_inputs']
        shapes = dict(observations=(e, 2, 11, 10), observation_mask=(e, 2, 11),
            observation_sources=(e, 2), forecasts=(e, c, 6, 6, 5),
            forecast_mask=(e, c, 6, 6), forecast_context=(e, c, 2),
            entity_mask=(e,))
        for name in ('observations', 'forecasts', 'forecast_context'):
            tensors[name].append(_numeric_tensor(raw[name], shapes[name], device, name))
        for name in ('observation_mask', 'forecast_mask', 'entity_mask'):
            tensors[name].append(_bool_tensor(raw[name], shapes[name], device, name))
        source = _numeric_tensor(raw['observation_sources'], shapes['observation_sources'],
                                 device, 'observation_sources')
        tensors['observation_sources'].append(source.to(torch.long))

        motion = prepared['ego_motion']
        speed, yaw_rate = motion['speed_mps'], motion['yaw_rate_rps']
        motions.append(torch.tensor([0. if speed is None else speed / spec.position_scale_m,
            0. if yaw_rate is None else yaw_rate * spec.time_scale_s,
            float(speed is not None), float(yaw_rate is not None)],
            device=device, dtype=torch.float32))
        _feature_history(features, prepared)
        history = torch.zeros((11, 4), device=device)
        history_mask = torch.zeros(11, dtype=torch.bool, device=device)
        if prepared['ego_history_used'] is not None:
            value = prepared['ego_history_used']
            states = torch.as_tensor(value['states'], device=device, dtype=torch.float32)
            times = torch.as_tensor(value['times'], device=device, dtype=torch.float32)
            history[:, :2] = states[:, :2] / spec.position_scale_m
            history[:, 2] = states[:, 2]
            history[:, 3] = times / spec.time_scale_s
            history_mask = torch.as_tensor(value['valid'], device=device, dtype=torch.bool)
        histories.append(history)
        history_masks.append(history_mask)
        previous = prepared['previous_plan']
        prior_plans.append(torch.zeros((6, 2), device=device) if previous is None else
                           torch.as_tensor(previous, device=device, dtype=torch.float32))
        prior_valid.append(torch.full((6,), previous is not None,
                                      device=device, dtype=torch.bool))
    batch = {name: torch.stack(values) for name, values in tensors.items()}
    batch.update(scene_patches=torch.stack(scene_patches), scene_mask=torch.stack(scene_masks),
        ego_motion=torch.stack(motions), ego_history=torch.stack(histories),
        ego_history_mask=torch.stack(history_masks), previous_plan=torch.stack(prior_plans),
        previous_plan_valid=torch.stack(prior_valid))
    return batch


def _masked_point_max(values, mask, encoder, null):
    values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    encoded = encoder(values)
    floor = torch.finfo(encoded.dtype).min
    pooled = encoded.masked_fill(~mask.unsqueeze(-1), floor).amax(dim=-2)
    valid = mask.any(dim=-1)
    return torch.where(valid.unsqueeze(-1), pooled, null.expand_as(pooled)), valid


class StructuredPlannerNetwork(nn.Module):
    """One shared trajectory network for Ego, P, F and combined evidence."""
    def __init__(self, spec):
        super().__init__()
        if not isinstance(spec, StructuredDriverSpec):
            raise ValueError('StructuredDriverSpec instance required')
        self.spec = spec
        d, heads = spec.hidden_dim, spec.attention_heads
        self.scene_projection = nn.Sequential(nn.Linear(320, d), nn.LayerNorm(d), nn.GELU())
        self.frame_code = nn.Parameter(torch.empty(2, d))
        self.grid_row_code = nn.Parameter(torch.empty(10, d))
        self.grid_column_code = nn.Parameter(torch.empty(22, d))
        self.observation_encoder = nn.Sequential(nn.Linear(10, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        self.observation_post = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        self.source_role = nn.Embedding(2, d)
        self.source_query = nn.Parameter(torch.empty(1, d))
        self.null_observation = nn.Parameter(torch.empty(1, d))
        self.source_attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
        self.source_norm = nn.LayerNorm(d)
        self.forecast_encoder = nn.Sequential(nn.Linear(5, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        self.forecast_context = nn.Sequential(nn.Linear(2, d), nn.LayerNorm(d))
        self.null_forecast = nn.Parameter(torch.empty(1, d))
        self.ego_history_encoder = nn.Sequential(nn.Linear(4, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        self.null_ego_history = nn.Parameter(torch.empty(1, d))
        self.motion_encoder = nn.Sequential(nn.Linear(4, d), nn.LayerNorm(d), nn.GELU())
        self.null_memory = nn.Parameter(torch.empty(1, d))
        self.waypoint_queries = nn.Parameter(torch.empty(6, d))
        self.prior_encoder = nn.Sequential(nn.Linear(2, d), nn.LayerNorm(d), nn.GELU(),
                                           nn.Linear(d, d))
        self.missing_prior = nn.Parameter(torch.empty(6, d))
        self.interactions = nn.ModuleList([nn.TransformerDecoderLayer(d, heads,
            dim_feedforward=2 * d, dropout=0., activation='gelu', batch_first=True,
            norm_first=True) for _ in range(spec.interaction_layers)])
        self.output_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                         nn.Linear(d, 2))
        self.reset_parameters()

    def reset_parameters(self):
        for value in (self.frame_code, self.grid_row_code, self.grid_column_code,
                      self.source_query, self.null_observation, self.null_forecast,
                      self.null_ego_history, self.null_memory, self.waypoint_queries,
                      self.missing_prior):
            nn.init.normal_(value, std=.02)

    def _validate_batch(self, batch):
        if not isinstance(batch, dict) or set(batch) != set(_BATCH_KEYS):
            raise ValueError('invalid structured model batch fields')
        b = batch['scene_patches'].shape[0]
        e, c = self.spec.max_entities, self.spec.max_forecast_sets_per_entity
        expected = dict(scene_patches=(b, 2, 220, 320), scene_mask=(b, 2, 220),
            observations=(b, e, 2, 11, 10), observation_mask=(b, e, 2, 11),
            observation_sources=(b, e, 2), forecasts=(b, e, c, 6, 6, 5),
            forecast_mask=(b, e, c, 6, 6), forecast_context=(b, e, c, 2),
            entity_mask=(b, e), ego_motion=(b, 4), ego_history=(b, 11, 4),
            ego_history_mask=(b, 11), previous_plan=(b, 6, 2),
            previous_plan_valid=(b, 6))
        for name, shape in expected.items():
            if not isinstance(batch[name], torch.Tensor) or tuple(batch[name].shape) != shape:
                raise ValueError('invalid structured model tensor: ' + name)
        return b

    def forward(self, batch):
        b = self._validate_batch(batch)
        scene_values = torch.where(batch['scene_mask'].unsqueeze(-1),
                                   batch['scene_patches'],
                                   torch.zeros_like(batch['scene_patches']))
        scene = self.scene_projection(scene_values)
        grid = (self.grid_row_code[:, None] + self.grid_column_code[None, :]).reshape(220, -1)
        scene = scene + self.frame_code[None, :, None] + grid[None, None]
        scene = scene.reshape(b, 440, -1)
        scene_valid = batch['scene_mask'].reshape(b, 440)

        obs, source_valid = _masked_point_max(batch['observations'],
            batch['observation_mask'], self.observation_encoder, self.null_observation)
        obs = self.observation_post(obs) + self.source_role(batch['observation_sources'])
        flat_obs = obs.reshape(b * self.spec.max_entities, 2, -1)
        flat_valid = source_valid.reshape(b * self.spec.max_entities, 2)
        safe_valid = flat_valid.clone()
        safe_valid[:, 0] |= ~safe_valid.any(dim=1)
        null = self.null_observation.reshape(1, 1, -1)
        flat_obs = torch.where(flat_valid.unsqueeze(-1), flat_obs, null)
        query = self.source_query.reshape(1, 1, -1).expand(len(flat_obs), 1, -1)
        fused, _ = self.source_attention(query, flat_obs, flat_obs,
                                         key_padding_mask=~safe_valid, need_weights=False)
        state = self.source_norm(query + fused).reshape(b, self.spec.max_entities, -1)

        future, future_valid = _masked_point_max(batch['forecasts'],
            batch['forecast_mask'], self.forecast_encoder, self.null_forecast)
        future = future + self.forecast_context(batch['forecast_context']).unsqueeze(-2)
        future_valid = future_valid & batch['entity_mask'][:, :, None, None]
        future = future.reshape(b, -1, self.spec.hidden_dim)
        future_valid = future_valid.reshape(b, -1)

        ego, ego_valid = _masked_point_max(batch['ego_history'], batch['ego_history_mask'],
            self.ego_history_encoder, self.null_ego_history)
        motion = self.motion_encoder(batch['ego_motion'])
        null_memory = self.null_memory.expand(b, 1, -1)
        memory = torch.cat([scene, state, future, ego[:, None], motion[:, None], null_memory], dim=1)
        memory_valid = torch.cat([scene_valid, batch['entity_mask'], future_valid,
            torch.ones((b, 3), dtype=torch.bool, device=scene.device)], dim=1)

        previous_valid = batch['previous_plan_valid']
        prior = self.prior_encoder(batch['previous_plan'] / self.spec.position_scale_m)
        prior = torch.where(previous_valid.unsqueeze(-1), prior,
                            self.missing_prior.unsqueeze(0))
        decoded = self.waypoint_queries.unsqueeze(0) + prior
        for layer in self.interactions:
            decoded = layer(decoded, memory, memory_key_padding_mask=~memory_valid)
        residual = self.output_head(decoded) * self.spec.position_scale_m
        base = torch.where(previous_valid.unsqueeze(-1), batch['previous_plan'],
                           torch.zeros_like(batch['previous_plan']))
        return base + residual


def _validate_model_version(value):
    if (not isinstance(value, dict) or set(value) != {'name', 'revision', 'training'} or
            not all(isinstance(value[key], str) and value[key] for key in ('name', 'revision')) or
            not isinstance(value['training'], dict) or
            not isinstance(value['training'].get('status'), str) or
            not value['training'].get('status') or
            type(value['training'].get('optimizer_steps')) is not int or
            value['training']['optimizer_steps'] < 0):
        raise ValueError('stable model version and training metadata required')
    _version_pair({key: value[key] for key in ('name', 'revision')})
    _check_json_native(value)
    return copy.deepcopy(value)


def _numeric_token_count(batch):
    """Count unmasked final-attention tokens plus the six waypoint queries."""
    forecast_modes = batch['forecast_mask'].any(dim=-1).sum()
    return int((batch['scene_mask'].sum() + batch['entity_mask'].sum() +
                forecast_modes + 9).item())


class StructuredPlanner:
    def __init__(self, spec, *, model=None, device='cpu', model_version=None):
        if not isinstance(spec, StructuredDriverSpec):
            raise ValueError('StructuredDriverSpec instance required')
        self.spec, self.device = spec, torch.device(device)
        self.model = StructuredPlannerNetwork(spec) if model is None else model
        if (not isinstance(self.model, StructuredPlannerNetwork) or
                self.model.spec.to_dict() != spec.to_dict()):
            raise ValueError('numeric planner model/spec mismatch')
        self.model.to(self.device)
        version = _validate_model_version(DEFAULT_MODEL_VERSION if model_version is None
                                          else model_version)
        self.provenance = dict(driver_kind='structured', driver_spec=spec.to_dict(),
                               decoding='numeric', model_version=version)

    def prepare_input(self, features, motion, ledger, receiver_spec,
                      previous_plan=None, previous_parent_refs=()):
        if isinstance(receiver_spec, dict):
            receiver_spec = StructuredDriverSpec.from_dict(receiver_spec)
        if not isinstance(receiver_spec, StructuredDriverSpec) or receiver_spec != self.spec:
            raise ValueError('runtime receiver and numeric planner specifications differ')
        history = None
        names = ('ego_pose_history', 'ego_pose_history_valid', 'ego_pose_history_times')
        if any(name in features for name in names):
            if not all(name in features for name in names):
                raise ValueError('ego pose history arrays must be supplied together')
            converted = []
            for name in names:
                value = features[name]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                converted.append(np.asarray(value))
            states, valid, times = converted
            if (states.shape != (11, 3) or states.dtype.kind not in 'fiu' or
                    valid.shape != (11,) or valid.dtype != np.dtype(bool) or
                    times.shape != (11,) or times.dtype.kind not in 'fiu' or
                    not np.isfinite(states).all() or not np.isfinite(times).all()):
                raise ValueError('invalid ego pose history feature layout')
            history = dict(states=states.astype(float).tolist(), valid=valid.tolist(),
                           times=times.astype(float).tolist(), read_paths=[])
        return build_structured_plan_input(motion, ledger, receiver_spec,
            ego_history=history, previous_plan=previous_plan,
            previous_parent_refs=previous_parent_refs)

    def plan_prepared(self, features, prepared):
        begin = perf_counter()
        batch = collate_structured_inputs([features], [prepared], self.device)
        if prepared['driver_spec'] != self.spec.to_dict():
            raise ValueError('prepared input and planner specifications differ')
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(batch)
        if tuple(prediction.shape) != (1, 6, 2) or not torch.isfinite(prediction).all().item():
            raise ValueError('numeric planner returned invalid waypoints')
        waypoints = prediction[0].detach().cpu().tolist()
        return dict(output_version='toolv2x_numeric_plan_v1', driver_kind='structured',
            status='valid', waypoints=waypoints, prepared_input=copy.deepcopy(prepared),
            driver_cost=dict(seconds=perf_counter() - begin,
                numeric_token_count=_numeric_token_count(batch), output_points=6,
                model_executed=True),
            parent_refs=copy.deepcopy(prepared['admission_report']['admitted_field_refs']))

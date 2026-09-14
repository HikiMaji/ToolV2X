"""Torch-free validation for numeric structured-driver outputs."""
import math

from planning.structured_inputs import validate_structured_prepared
from tools.task_spec import _keys, validate_plan


def validate_numeric_output(output, prepared, execution_spec):
    """Return the six actual numeric waypoints after strict output validation."""
    _keys(output, ('output_version', 'driver_kind', 'status', 'waypoints',
                   'prepared_input', 'driver_cost', 'parent_refs'),
          'numeric driver output')
    if (output['output_version'] != 'toolv2x_numeric_plan_v1' or
            output['driver_kind'] != 'structured' or output['status'] != 'valid'):
        raise ValueError('invalid numeric driver output identity')
    validate_structured_prepared(prepared)
    validate_structured_prepared(output['prepared_input'])
    if output['prepared_input'] != prepared:
        raise ValueError('numeric output does not bind the exact prepared input')
    expected_refs = prepared['admission_report']['admitted_field_refs']
    if output['parent_refs'] != expected_refs:
        raise ValueError('numeric output parent closure changed')
    cost = output['driver_cost']
    _keys(cost, ('seconds', 'numeric_token_count', 'output_points', 'model_executed'),
          'numeric driver cost')
    seconds = cost['seconds']
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0 or
            type(cost['numeric_token_count']) is not int or cost['numeric_token_count'] <= 0 or
            cost['output_points'] != 6 or cost['model_executed'] is not True):
        raise ValueError('invalid numeric driver cost')
    return validate_plan(output['waypoints'], execution_spec)

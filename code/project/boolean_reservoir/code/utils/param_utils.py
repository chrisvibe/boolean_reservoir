from pydantic import BaseModel, model_validator, ConfigDict, Field
import yaml
from itertools import product
from typing import Callable, ClassVar
from pathlib import Path, PosixPath, WindowsPath
import sympy
from inspect import signature, Parameter
import re

def pydantic_init():
    def represent_pathlib_path(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

    yaml.add_representer(Path, represent_pathlib_path)
    yaml.add_representer(PosixPath, represent_pathlib_path)
    yaml.add_representer(WindowsPath, represent_pathlib_path)

def calculate_w_broadcasting(operator: Callable[[float, float], float], a, b):
    # list-list, list-value, value-list, value-value
    if isinstance(a, list) and isinstance(b, list):
        return [operator(a[i], b[i]) for i in range(len(a))]
    elif isinstance(a, list):
        return [operator(x, b) for x in a]
    elif isinstance(b, list):
        return [operator(a, y) for y in b]
    else:
        return operator(a, b)


class ExpressionEvaluator:
    def __init__(self, symbols: dict = None):
        self.symbols = symbols or {}
        self._sympy_symbols = {k: sympy.Symbol(k) for k in self.symbols.keys()}

    def eval(self, expr):
        """Convert a string expression to a float using sympy."""
        if isinstance(expr, list):
            return expr 
        elif not isinstance(expr, str):
            return expr

        try:
            parsed_expr = sympy.sympify(expr)
            if self.symbols:
                symbol_values = {
                    self._sympy_symbols[k]: v for k, v in self.symbols.items()
                }
                result = parsed_expr.subs(symbol_values)
            else:
                result = parsed_expr
            return float(result.evalf())
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {e}")

class CallParams(BaseModel):
    """Base class for call parameters - extend this for specific use cases"""
    model_config = ConfigDict(extra='allow')  # Allow any field to be added

class DynamicParams(BaseModel):
    name: str
    params: CallParams = Field(default_factory=CallParams)

    def call(self, func, evaluator=None, **overrides):
        """
        Call a function using parameters from this DynamicParams instance, optionally
        evaluating string expressions.
        Args:
            func (callable): The target function to call.
            evaluator (ExpressionEvaluator, optional): If provided, string values in 
                self.params will be evaluated using this evaluator.
            **overrides: Optional parameter overrides for this call.
        Returns:
            The result of func(**final_params), where final_params is a merge of:
            - defaults from func signature
            - values from self.params (evaluated if evaluator provided)
            - explicit overrides in **overrides (unevaluated)
        Notes:
            - self.params remains unmodified; evaluation occurs only during the call.
            - Only parameters that exist in func's signature are passed.
            - Only self.params values are evaluated; overrides are passed as-is.
            - Priority order: defaults < self.params < overrides.
        """
        sig = signature(func)
        # Evaluate self.params that are in func signature
        evaluated = {k: evaluator.eval(v) if evaluator else v
                     for k, v in self.params if k in sig.parameters}
        # Overrides go in unevaluated, highest priority
        valid_params = {**evaluated, **overrides}
        # Get defaults from function signature
        defaults = {n: p.default for n, p in sig.parameters.items()
                    if p.default is not Parameter.empty}
        # Merge defaults with valid params
        final_params = {**defaults, **valid_params}
        return func(**final_params)

    _evaluator: ClassVar[ExpressionEvaluator] = ExpressionEvaluator()

    @model_validator(mode='after')
    def evaluate_expressions(self):
        """Evaluate any expressions in params"""
        def evaluate_value(x):
            if isinstance(x, str):
                try:
                    return self._evaluator.eval(x)
                except:
                    return x
            elif isinstance(x, list):
                return [evaluate_value(item) for item in x]
            return x
        
        params_dict = self.params.model_dump()
        self.params = CallParams(**{key: evaluate_value(value) for key, value in params_dict.items()})
        return self 

def generate_param_combinations(params):
    if not isinstance(params, list):
        params = [params]
    all_combinations = []
    for param in params:
        if isinstance(param, BaseModel):
            params_dict = {}
            all_fields = param.model_dump()
            for field_name in all_fields.keys():
                value = getattr(param, field_name)
                field_info = param.model_fields.get(field_name)
                should_expand = True
                if field_info and field_info.json_schema_extra:
                    should_expand = field_info.json_schema_extra.get('expand', True)
                if not should_expand:
                    params_dict[field_name] = [value]
                else:
                    params_dict[field_name] = generate_param_combinations(value)
            all_combinations.extend(
                _generate_combinations_from_dict(params_dict, param.__class__)
            )
        elif isinstance(param, dict):
            expanded = {k: generate_param_combinations(v) for k, v in param.items()}
            all_combinations.extend(
                _generate_combinations_from_dict(expanded, dict)
            )
        elif isinstance(param, list) and not any(isinstance(x, (BaseModel, dict)) for x in param):
            all_combinations.append(param)
        else:
            all_combinations.append(param)
    return all_combinations

def _generate_combinations_from_dict(expanded_fields, original_type):
    field_names = list(expanded_fields.keys())
    field_values = expanded_fields.values()
    combos = []
    for combo in product(*field_values):
        if original_type == list:
            combos.append(list(combo))
        elif original_type == dict:
            combos.append(dict(zip(field_names, combo)))
        else:
            combos.append(original_type(**dict(zip(field_names, combo))))
    return combos

def expand_ticks(s):
    # Expand pattern repetition: '(123){3}' → '123123123'
    s = re.sub(r'\(([^)]+)\)\{(\d+)\}', lambda m: m.group(1) * int(m.group(2)), s)
    # Expand single-char RLE: '1{3}' → '111'
    s = re.sub(r'(.)\{(\d+)\}', lambda m: m.group(1) * int(m.group(2)), s)
    return s
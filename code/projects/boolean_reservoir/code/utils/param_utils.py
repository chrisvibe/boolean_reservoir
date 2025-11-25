from pydantic import BaseModel, Field, field_validator
import yaml
from itertools import product
from typing import List, Callable, Dict, Any, ClassVar
from pathlib import Path, PosixPath, WindowsPath
import sympy
from inspect import signature, Parameter

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

def generate_param_combinations(params):
    if not isinstance(params, list):
        params = [params]

    all_combinations = []
    for param in params:
        if isinstance(param, BaseModel):
            params_dict = {}
            for field_name, field_info in param.model_fields.items():
                value = getattr(param, field_name)
                if field_info.json_schema_extra and field_info.json_schema_extra.get('expand', True) is False:
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
            return param
        else:
            all_combinations.append(param)
    return all_combinations


def _generate_combinations_from_dict(expanded_fields, original_type):
    import itertools
    field_names = list(expanded_fields.keys())
    field_values = expanded_fields.values()
    combos = []
    for combo in itertools.product(*field_values):
        if original_type == list:
            combos.append(list(combo))
        elif original_type == dict:
            combos.append(dict(zip(field_names, combo)))
        else:
            combos.append(original_type(**dict(zip(field_names, combo))))
    return combos


class ExpressionEvaluator:
    def __init__(self, symbols: dict = None):
        self.symbols = symbols or {}
        self._sympy_symbols = {k: sympy.Symbol(k) for k in self.symbols.keys()}

    def eval(self, expr):
        """Convert a string expression to a float using sympy."""
        if isinstance(expr, (int, float)):
            return expr
        if not isinstance(expr, str):
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
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {e}")


class DynamicParams(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)

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
                     for k, v in self.params.items() if k in sig.parameters}
        # Overrides go in unevaluated, highest priority
        valid_params = {**evaluated, **overrides}
        # Get defaults from function signature
        defaults = {n: p.default for n, p in sig.parameters.items()
                    if p.default is not Parameter.empty}
        # Merge defaults with valid params
        final_params = {**defaults, **valid_params}
        return func(**final_params)

    _evaluator: ClassVar[ExpressionEvaluator] = ExpressionEvaluator()
    
    @field_validator('params')
    @classmethod
    def evaluate_expressions(cls, v):
        """Evaluate any expressions in params"""
        def evaluate_value(x):
            if isinstance(x, str):
                try:
                    return cls._evaluator.eval(x)  # Now this works
                except:
                    return x
            elif isinstance(x, list):
                return [evaluate_value(item) for item in x]
            return x
        
        return {key: evaluate_value(value) for key, value in v.items()}
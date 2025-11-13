from pydantic import BaseModel, Field
import yaml
from itertools import product
from typing import List, Callable, Dict, Any
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

def generate_param_combinations(params: BaseModel) -> List[BaseModel]:
    def expand_params(params: Any) -> List[Dict[str, Any]]:
        if isinstance(params, BaseModel):
            params_dict = params.__dict__
            expandable = {}
            for k in params_dict.keys():
                field_info = type(params).model_fields.get(k)
                expand = True
                if field_info and field_info.json_schema_extra is not None:
                    expand = field_info.json_schema_extra.get('expand', True)
                expandable[k] = expand
        elif isinstance(params, dict):
            params_dict = params
            expandable = {k: True for k in params_dict.keys()}
        elif isinstance(params, list):
            return params
        else:
            return [params]
        
        expanded_dict = {}
        for k, v in params_dict.items():
            if expandable[k]:
                expanded_dict[k] = expand_params(v)
            else:
                expanded_dict[k] = [v]
        
        keys = expanded_dict.keys()
        values = [expanded_dict[k] for k in keys]
        combinations = product(*values)
        return [dict(zip(keys, combo)) for combo in combinations]

    expanded_params = expand_params(params)
    result_params_list = []

    for combo_dict in expanded_params:
        new_params_dict = {k: type(params.__dict__[k])(**combo_dict[k]) if isinstance(params.__dict__[k], BaseModel) else combo_dict[k] for k in params.__dict__.keys()}
        result_params_list.append(type(params)(**new_params_dict))
    
    return result_params_list

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
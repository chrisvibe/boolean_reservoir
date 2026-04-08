import hashlib
import json
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
from project.boolean_reservoir.code.parameter import Params, load_yaml_config, deep_merge
from enum import Enum
from inspect import getsource
from typing import get_origin, get_args, Union, Type, Optional
import re

class _ParamsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def save_grid_search_results(df: pd.DataFrame, path: Path):
    """Append grid search results to a Parquet file."""
    path = Path(path).with_suffix('.parquet')
    json_blobs = [json.dumps(p.model_dump(), cls=_ParamsEncoder) for p in df['params']]
    new_table = pa.table({'params_json': json_blobs})

    if path.exists():
        existing = pq.read_table(path)
        new_table = pa.concat_tables([existing, new_table])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(new_table, path)

class DotDict(dict):
    __slots__ = ('_tree', '_cls', '_merge_base')

    def __init__(self, data, cls=None, tree=None, _merge_base=None):
        super().__init__(data)
        object.__setattr__(self, '_cls', cls)
        object.__setattr__(self, '_merge_base', _merge_base)
        object.__setattr__(self, '_tree', tree or (DotDict._alias_tree(cls) if cls else {}))

    def __getattr__(self, key):
        merge_base = object.__getattribute__(self, '_merge_base')
        if merge_base is not None:
            # Universe view: key is the universe name; merge its overrides into base
            try:
                override = dict.__getitem__(self, key)
            except KeyError:
                return merge_base  # unknown universe → return base unchanged
            cls = object.__getattribute__(self, '_cls')
            return DotDict(deep_merge(dict(merge_base), override), cls=cls)

        if key == 'U':
            cls = object.__getattribute__(self, '_cls')
            overrides = dict.get(self, 'multiverse_overrides') or {}
            return DotDict(overrides, cls=cls, _merge_base=self)

        tree = object.__getattribute__(self, '_tree')
        resolved = tree.get('a', {}).get(key, key)
        try:
            val = self[resolved]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, dict):
            child_tree = tree.get('c', {}).get(resolved)
            return DotDict(val, tree=child_tree)
        return val

    _tree_cache: dict = {}

    @staticmethod
    def _alias_tree(cls: Type[BaseModel]) -> dict:
        """Build {'a': {alias: field}, 'c': {field: subtree}} for cls and children."""
        if cls in DotDict._tree_cache:
            return DotDict._tree_cache[cls]

        aliases = {}
        for name, obj in vars(cls).items():
            if isinstance(obj, property) and obj.fget:
                m = re.search(r'return self\.(\w+)\s*$', getsource(obj.fget), re.MULTILINE)
                if m:
                    aliases[name] = m.group(1)

        children = {}
        for fname, finfo in cls.model_fields.items():
            ann = finfo.annotation
            if get_origin(ann) is Union:
                ann = next((a for a in get_args(ann) if a is not type(None)), ann)
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                children[fname] = DotDict._alias_tree(ann)

        result = {'a': aliases, 'c': children} if aliases or children else {}
        DotDict._tree_cache[cls] = result
        return result
    
    def to_pydantic(self):
        cls = object.__getattribute__(self, '_cls')
        if cls is None:
            raise ValueError("No Pydantic class associated")
        return cls.model_validate(dict(self))

def load_params_df(
    data_path: Path,
    model_class: Type[BaseModel] = Params,
    fast: bool = True,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    keep_params_json: bool = False,
) -> pd.DataFrame:
    """Load Parquet → DataFrame with hydrated 'params' column.
    
    Args:
        data_path: Path to the parquet file.
        model_class: Pydantic model class for validation.
        fast: If True, use DotDict (skip Pydantic validation) for faster loading.
              If False, use full Pydantic model_validate.
        limit: Maximum number of rows to return.
        batch_size: Number of rows to read per batch.
    """
    data_path = Path(data_path).with_suffix('.parquet')
    
    # 1. Load data with optional row limit and batching
    if limit is not None or batch_size is not None:
        pf = pq.ParquetFile(data_path)
        batches = []
        rows_read = 0
        
        # Smart default: If no batch_size is provided, use the limit (if set) 
        # to prevent fetching more rows than necessary. Otherwise fallback to PyArrow's default.
        _batch_size = batch_size if batch_size is not None else (limit if limit is not None else 65536)
        
        for batch in pf.iter_batches(batch_size=_batch_size):
            batches.append(batch)
            rows_read += batch.num_rows
            if limit is not None and rows_read >= limit:
                break
                
        if batches:
            table = pa.Table.from_batches(batches)
            if limit is not None:
                table = table.slice(length=limit)
            df = table.to_pandas()
        else:
            df = pd.DataFrame()
            
    else:
        # Fast path if we just want the whole file in memory
        df = pq.read_table(data_path).to_pandas()

    # 2. Process the 'params_json' column
    if not df.empty and 'params_json' in df.columns:
        if fast:
            df['params'] = df['params_json'].apply(
                lambda s: DotDict(orjson.loads(s), cls=Params)
            )
        else:
            df['params'] = df['params_json'].apply(
                lambda s: model_class.model_validate(orjson.loads(s))
            )
        if not keep_params_json:
            df.drop(columns=['params_json'], inplace=True)
        
    return df

def params_col_to_fields(df, extractions):
    """
    Projects structured parameter objects in `df['params']`
    into a new DataFrame of extracted fields.
    Args:
        df: DataFrame containing a `params` column.
        extractions: List of (prefix, getter, field_set) tuples.
            - prefix: Column prefix or column name if capturing source.
            - getter: Function extracting a sub-model from params.
            - field_set: Set of field names to extract, empty set {} for all fields,
                        or None to capture source object.
    Returns:
        (new_df, factors):
            new_df: DataFrame with extracted fields (and captured sources).
            factors: List of extracted flattened column names.
    """
    rows = []
    factors = []
    for params in df['params']:
        row = {}
        for prefix, get_source, field_set in extractions:
            source = get_source(params)
            if source is None:
                lambda_str = getsource(get_source).strip() 
                print(f"Warning: Extraction source is None for extraction: {lambda_str}")
                continue
            if field_set is None:
                row[prefix] = source
                if not isinstance(source, (dict, BaseModel)) and prefix not in factors:
                    factors.append(prefix)
                continue
            
            dumped = source if isinstance(source, dict) else source.model_dump()
            # If field_set is empty, extract all fields
            fields_to_extract = dumped.keys() if not field_set else field_set
            
            for k in fields_to_extract:
                v = dumped.get(k)
                col = f"{prefix}_{k}"
                row[col] = str(v) if isinstance(v, Enum) else v
                if col not in factors:
                    factors.append(col)
        rows.append(row)
    return pd.DataFrame(rows), factors

def get_data_path(config_path, filename='log.parquet') -> Path:
    """Derive data path from config's out_path"""
    P = load_yaml_config(config_path)
    return P.L.out_path / filename

def _cache_key(data_paths, extractions, df_filter_mask, filename, limit, batch_size, keep_params_json) -> str:
    extraction_sources = [getsource(fn) for _, fn, _ in extractions] if extractions else []
    mask_source = getsource(df_filter_mask) if df_filter_mask else ''
    file_sigs = [(str(p), p.stat().st_mtime, p.stat().st_size) for p in data_paths]
    key_material = str((extraction_sources, mask_source, file_sigs, filename, limit, batch_size, keep_params_json))
    return hashlib.md5(key_material.encode()).hexdigest()

def custom_load_grid_search_data(data_paths=None, config_paths=None, extractions=None, df_filter_mask=None, filename='log.parquet', limit=None, batch_size=None, keep_params_json: bool = True, cache_dir: Optional[Path | str] = Path('/tmp/boolean_reservoir/cache/custom_load_grid_search_data')) -> tuple[pd.DataFrame, list[str]]:
    """Core loader - requires explicit data_paths or config_paths + extractions.

    Args:
        cache_dir: Directory for caching extracted DataFrames. Pass None to disable caching.
    """
    if data_paths is None and config_paths is None:
        raise ValueError("Must provide data_paths or config_paths")

    if data_paths is None:
        if isinstance(config_paths, (str, Path)):
            config_paths = [config_paths]
        data_paths = [get_data_path(p, filename) for p in config_paths]

    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]

    data_paths = [Path(p) for p in data_paths]

    if cache_dir is not None and extractions:
        cache_dir = Path(cache_dir)
        cache_key = _cache_key(data_paths, extractions, df_filter_mask, filename, limit, batch_size, keep_params_json)
        cache_path = cache_dir / f'{cache_key}.parquet'
        factors_path = cache_dir / f'{cache_key}.factors.json'
        if cache_path.exists() and factors_path.exists():
            print(f'Loading from cache: {cache_path}')
            df = pd.read_parquet(cache_path)
            factors = json.loads(factors_path.read_text())
            return df, factors

    dfs = []
    factors = []
    for path in data_paths:
        df_raw = load_params_df(data_path=path, limit=limit, batch_size=batch_size, keep_params_json=keep_params_json)
        params_json_col = df_raw['params_json'].reset_index(drop=True) if 'params_json' in df_raw.columns else None
        if extractions:
            df, factors = params_col_to_fields(df_raw, extractions)
            if params_json_col is not None:
                df['params_json'] = params_json_col
        else:
            df = df_raw
        if df_filter_mask:
            df = df[df_filter_mask(df)]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    if cache_dir is not None and extractions:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        factors_path.write_text(json.dumps(factors))
        print(f'Cached to: {cache_path}')

    return df, factors


def load_grid_search_data(data_paths=None, config_paths=None, extractions=None, df_filter_mask=None, filename='log.parquet', keep_params_json: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Convenience loader with default train_log extraction"""
    if extractions is None:
        extractions = [
            ('P', lambda p: p, None),
            ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
        ]

    return custom_load_grid_search_data(data_paths=data_paths, config_paths=config_paths, extractions=extractions, df_filter_mask=df_filter_mask, filename=filename, keep_params_json=keep_params_json)
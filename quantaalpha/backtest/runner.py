#!/usr/bin/env python3
"""
Backtest runner using Qlib: load factors (official/custom), compute custom factor values, train, backtest, evaluate.
Modes: official (Qlib DataLoader) or custom (expr_parser + function_lib).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Backtest executor."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._qlib_initialized = False
        # 数据频率（day / 5min / 1min）。原来全部硬编码为 'day'，
        # 分钟级数据集跑不起来；改成从配置读，默认仍是 day，日频流程行为不变。
        self.freq = self._resolve_freq()
        # 同步给自定义因子计算器（它从 config['data'] 读参数）
        self.config.setdefault('data', {})['freq'] = self.freq
        self.ann_scaler = self._resolve_ann_scaler()
        logger.info(f"Backtest freq: {self.freq} (ann_scaler={self.ann_scaler})")

    # 每年 bar 数：日频 252；5 分钟 252*48；1 分钟 252*240
    _BARS_PER_YEAR = {"day": 252, "5min": 252 * 48, "1min": 252 * 240}

    def _resolve_ann_scaler(self) -> int:
        """年化系数：配置里可显式给 ann_scaler，否则按频率折算。"""
        for section in ("backtest", "data", "qlib_init", ""):
            src = self.config.get(section, self.config) if section else self.config
            if isinstance(src, dict) and src.get("ann_scaler"):
                return int(src["ann_scaler"])
        return int(self._BARS_PER_YEAR.get(self.freq, 252))

    def _resolve_freq(self) -> str:
        """解析数据集频率，优先级：data.freq > qlib_init.freq > data_loader.kwargs.freq > 'day'。

        三种写法分别对应：configs/backtest.yaml 风格（data 段）、
        conf_baseline.yaml 风格（qlib_init 段或 data_loader 构造参数）。
        """
        freq = (self.config.get('data') or {}).get('freq')
        if not freq:
            freq = (self.config.get('qlib_init') or {}).get('freq')
        if not freq:
            loader = ((self.config.get('data_handler_config') or {}).get('data_loader') or {})
            freq = (loader.get('kwargs') or {}).get('freq')
        return freq or 'day'

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config: {self.config_path}")
        return config

    def _init_qlib(self):
        if self._qlib_initialized:
            return
        import os
        import qlib

        cfg_data = self.config.get('data') or {}
        # 数据集级 override：配置显式声明 data.force_provider_uri=true 时忽略环境变量。
        # 必要性：项目 .env 里的 QLIB_DATA_DIR 是"全局默认数据集"（日频 cn_real_full），
        # 且 env loader 会强制覆盖命令行传进来的同名变量 —— 想跑 5 分钟数据集时，
        # 在命令行设 QLIB_DATA_DIR 根本不起作用，必须让配置自己说了算。
        # 同时补上 qlib_init.provider_uri 兜底（原实现只认 data.provider_uri）。
        if cfg_data.get('force_provider_uri') and cfg_data.get('provider_uri'):
            provider_uri = cfg_data['provider_uri']
        else:
            provider_uri = (
                os.environ.get('QLIB_DATA_DIR')
                or os.environ.get('QLIB_PROVIDER_URI')
                or cfg_data.get('provider_uri')
                or (self.config.get('qlib_init') or {}).get('provider_uri')
            )
        provider_uri = os.path.expanduser(provider_uri)
        region = cfg_data.get('region', 'cn')
        qlib.init(provider_uri=provider_uri, region=region)
        self._qlib_initialized = True
        logger.info(f"Qlib initialized: {provider_uri} (region={region})")

    def run(self,
            factor_source: Optional[str] = None,
            factor_json: Optional[List[str]] = None,
            experiment_name: Optional[str] = None,
            output_name: Optional[str] = None,
            skip_uncached: bool = False) -> Dict:
        """Run full backtest; returns metrics dict."""
        start_time_total = time.time()
        self._init_qlib()
        if factor_source:
            self.config['factor_source']['type'] = factor_source
        if factor_json:
            self.config['factor_source']['custom']['json_files'] = factor_json
        
        if output_name is None and factor_json:
            output_name = Path(factor_json[0]).stem

        exp_name = experiment_name or output_name or self.config['experiment']['name']
        rec_name = self.config['experiment']['recorder']

        print(f"\n{'='*50}")
        src = factor_json[0] if factor_json else exp_name
        print(f"Starting backtest: {src}")
        print(f"{'='*50}")

        factor_expressions, custom_factors = self._load_factors()
        print(f"[1/4] Loaded factors: Qlib {len(factor_expressions)}, custom {len(custom_factors)}")

        computed_factors = None
        if custom_factors:
            computed_factors = self._compute_custom_factors(custom_factors, skip_compute=skip_uncached)
            n_computed = len(computed_factors.columns) if computed_factors is not None and not computed_factors.empty else 0
            print(f"[2/4] Computed custom factors: {n_computed}")
        else:
            logger.debug("[2/4] No custom factors, skip")

        dataset = self._create_dataset(factor_expressions, computed_factors)
        print("[3/4] Dataset created")

        metrics = self._train_and_backtest(dataset, exp_name, rec_name, output_name=output_name)
        total_time = time.time() - start_time_total
        self._print_results(metrics, total_time)
        self._save_results(metrics, exp_name, factor_source or self.config['factor_source']['type'], 
                          len(factor_expressions) + len(custom_factors), total_time,
                          output_name=output_name)
        
        return metrics
    
    def _load_factors(self) -> Tuple[Dict[str, str], List[Dict]]:
        from .factor_loader import FactorLoader
        
        loader = FactorLoader(self.config)
        return loader.load_factors()
    
    def _compute_custom_factors(self, factors: List[Dict], skip_compute: bool = False) -> Optional[pd.DataFrame]:
        """Compute custom factors (expr_parser + function_lib); supports cache; loads stock data only when needed."""
        from .custom_factor_calculator import CustomFactorCalculator
        from pathlib import Path

        llm_config = self.config.get('llm', {})
        cache_dir = llm_config.get('cache_dir')
        if cache_dir:
            cache_dir = Path(cache_dir)
        auto_extract = llm_config.get('auto_extract_cache', True)
        calculator = CustomFactorCalculator(
            data_df=None,
            cache_dir=cache_dir,
            auto_extract_cache=auto_extract,
            config=self.config,
        )
        result_df = calculator.calculate_factors_batch(factors, use_cache=True, skip_compute=skip_compute)
        if result_df is None:
            logger.error("Factor computation returned None")
            return None
        if not isinstance(result_df, pd.DataFrame):
            logger.error(f"Factor computation returned wrong type: {type(result_df)}")
            return None
        
        if result_df.empty:
            logger.error("Factor computation returned empty DataFrame")
            return None
        
        if not isinstance(result_df.index, pd.MultiIndex):
            logger.warning("Factor data index is not MultiIndex, attempting fix...")
        logger.debug(f"  Factor computation done: {len(result_df.columns)} factors, {len(result_df)} rows")
        
        return result_df
    
    def _create_dataset(self, 
                       factor_expressions: Dict[str, str],
                       computed_factors: Optional[pd.DataFrame] = None):
        """Create Qlib dataset (QlibDataLoader or precomputed factors + StaticDataLoader)."""
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        
        data_config = self.config['data']
        dataset_config = self.config['dataset']
        
        has_computed_factors = False
        if computed_factors is not None:
            if isinstance(computed_factors, pd.DataFrame):
                if len(computed_factors) > 0 and len(computed_factors.columns) > 0:
                    has_computed_factors = True
                    logger.debug(f"  Precomputed factors: {len(computed_factors.columns)} factors, {len(computed_factors)} rows")
                else:
                    logger.warning(f"  Precomputed factor DataFrame is empty: {computed_factors.shape}")
            else:
                logger.warning(f"  Precomputed factor type invalid: {type(computed_factors)}")
        
        # Prefer custom factor mode when computed factors exist
        if has_computed_factors:
            logger.debug("  Using custom factor mode (precomputed)")
            return self._create_dataset_with_computed_factors(
                factor_expressions, computed_factors
            )
        
        # Qlib-only factor mode
        expressions = list(factor_expressions.values())
        names = list(factor_expressions.keys())
        
        if not expressions:
            raise ValueError("No factor expressions available. If using custom factors, ensure factor computation succeeded.")
        
        handler_config = {
            'start_time': data_config['start_time'],
            'end_time': data_config['end_time'],
            'instruments': data_config['market'],
            'data_loader': {
                'class': 'QlibDataLoader',
                'module_path': 'qlib.contrib.data.loader',
                'kwargs': {
                    # 频率必须显式传入：QlibDataLoader 默认 freq='day'，
                    # 在分钟数据集上会去读 calendars/day.txt 导致取不到数据。
                    'freq': self.freq,
                    'config': {
                        'feature': (expressions, names),
                        'label': ([dataset_config['label']], ['LABEL0'])
                    }
                }
            },
            'learn_processors': dataset_config['learn_processors'],
            'infer_processors': dataset_config['infer_processors']
        }
        
        dataset = DatasetH(
            handler=DataHandlerLP(**handler_config),
            segments=dataset_config['segments']
        )
        
        logger.debug(f"  Qlib mode: {len(expressions)} factors, train={dataset_config['segments']['train']}")
        
        return dataset
    
    def _create_dataset_with_computed_factors(self,
                                              factor_expressions: Dict[str, str],
                                              computed_factors: pd.DataFrame):
        """Create dataset from precomputed factors: compute label, merge with factors, use custom DataHandler."""
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandler
        from qlib.data import D
        
        data_config = self.config['data']
        dataset_config = self.config['dataset']
        
        logger.debug(f"  Computed factor count: {len(computed_factors.columns)}")
        label_expr = dataset_config['label']
        label_df = self._compute_label(label_expr)
        
        all_feature_dfs = [computed_factors]
        if factor_expressions:
            logger.debug(f"  Loading {len(factor_expressions)} Qlib-compatible factors")
            qlib_factors = self._load_qlib_factors(factor_expressions)
            if qlib_factors is not None and not qlib_factors.empty:
                all_feature_dfs.append(qlib_factors)
        
        features_df = pd.concat(all_feature_dfs, axis=1)
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]
        logger.debug(f"  Total factor count: {len(features_df.columns)}")

        def _normalize_multiindex(df, df_name):
            """Ensure MultiIndex has standard (datetime, instrument) level names."""
            if not isinstance(df.index, pd.MultiIndex):
                logger.warning(f"  {df_name} index is not MultiIndex: {type(df.index)}")
                return df
            
            names = list(df.index.names)
            logger.debug(f"  {df_name} index levels: {names}, "
                        f"dtypes: {[str(df.index.get_level_values(i).dtype) for i in range(len(names))]}, "
                        f"len: {len(df)}")
            
            new_names = list(names)
            for i, name in enumerate(names):
                level_vals = df.index.get_level_values(i)
                if name == 'datetime' or name == 'date':
                    new_names[i] = 'datetime'
                elif name == 'instrument' or name == 'stock':
                    new_names[i] = 'instrument'
                elif name is None:
                    # Infer from dtype
                    if pd.api.types.is_datetime64_any_dtype(level_vals):
                        new_names[i] = 'datetime'
                    elif level_vals.dtype == object or pd.api.types.is_string_dtype(level_vals):
                        new_names[i] = 'instrument'
            
            if new_names != names:
                logger.debug(f"  {df_name} index renamed: {names} -> {new_names}")
                df.index = df.index.set_names(new_names)
            actual_names = list(df.index.names)
            if len(actual_names) == 2 and actual_names == ['instrument', 'datetime']:
                df = df.swaplevel()
                df = df.sort_index()
                logger.debug(f"  {df_name} index swapped to (datetime, instrument)")
            
            return df
        
        features_df = _normalize_multiindex(features_df, "features")
        label_df = _normalize_multiindex(label_df, "label")
        
        common_index = features_df.index.intersection(label_df.index)
        if len(common_index) == 0 and len(features_df) > 0 and len(label_df) > 0:
            logger.warning("  Index intersection empty, aligning datetime types...")
            feat_dt = features_df.index.get_level_values('datetime')
            label_dt = label_df.index.get_level_values('datetime')
            logger.debug(f"  features datetime dtype={feat_dt.dtype}, sample={feat_dt[:3].tolist()}")
            logger.debug(f"  label    datetime dtype={label_dt.dtype}, sample={label_dt[:3].tolist()}")
            
            feat_inst = features_df.index.get_level_values('instrument')
            label_inst = label_df.index.get_level_values('instrument')
            logger.debug(f"  features instrument sample={feat_inst[:3].tolist()}")
            logger.debug(f"  label    instrument sample={label_inst[:3].tolist()}")
            
            try:
                if not pd.api.types.is_datetime64_any_dtype(feat_dt):
                    features_df.index = features_df.index.set_levels(
                        pd.to_datetime(feat_dt.unique()), level='datetime'
                    )
                    logger.debug("  features datetime converted to Timestamp")
                if not pd.api.types.is_datetime64_any_dtype(label_dt):
                    label_df.index = label_df.index.set_levels(
                        pd.to_datetime(label_dt.unique()), level='datetime'
                    )
                    logger.debug("  label datetime converted to Timestamp")
            except Exception as e:
                logger.warning(f"  datetime type conversion failed: {e}")
            common_index = features_df.index.intersection(label_df.index)
            logger.debug(f"  Intersection size after align: {len(common_index)}")

        if len(common_index) == 0:
            logger.warning("  Index intersection still empty, trying merge...")
            feat_reset = features_df.reset_index()
            label_reset = label_df.reset_index()
            dt_col = 'datetime' if 'datetime' in feat_reset.columns else feat_reset.columns[0]
            inst_col = 'instrument' if 'instrument' in feat_reset.columns else feat_reset.columns[1]
            
            merged = pd.merge(
                feat_reset, label_reset,
                on=[dt_col, inst_col],
                how='inner'
            )
            logger.debug(f"  Merged rows: {len(merged)}")
            if len(merged) == 0:
                raise ValueError(
                    f"Factor and label data could not be aligned. "
                    f"features: {len(features_df)} rows, index names={list(features_df.index.names)}; "
                    f"label: {len(label_df)} rows, index names={list(label_df.index.names)}"
                )
            
            merged = merged.set_index([dt_col, inst_col])
            merged.index.names = ['datetime', 'instrument']
            
            feature_cols = [c for c in features_df.columns if c in merged.columns]
            label_cols = [c for c in label_df.columns if c in merged.columns]
            features_df = merged[feature_cols]
            label_df = merged[label_cols]
        else:
            features_df = features_df.loc[common_index]
            label_df = label_df.loc[common_index]
        
        logger.debug(f"  Data rows: {len(features_df)}")
        if len(features_df) == 0:
            raise ValueError("No rows after index alignment; cannot run backtest")
        combined_df = pd.concat([features_df, label_df], axis=1)
        from qlib.data.dataset.processor import Fillna, ProcessInf, CSRankNorm, DropnaLabel
        feature_cols = list(features_df.columns)
        label_cols = list(label_df.columns)
        combined_df[feature_cols] = combined_df[feature_cols].fillna(0)
        combined_df[feature_cols] = combined_df[feature_cols].replace([np.inf, -np.inf], 0)
        dt_level = combined_df.index.names[0] if combined_df.index.names[0] else 0
        for col in feature_cols:
            combined_df[col] = combined_df.groupby(level=dt_level)[col].transform(
                lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
            )
        combined_df = combined_df.dropna(subset=label_cols)
        for col in label_cols:
            combined_df[col] = combined_df.groupby(level=dt_level)[col].transform(
                lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
            )
        
        logger.debug(f"  Rows after preprocessing: {len(combined_df)}")
        feature_tuples = [('feature', col) for col in feature_cols]
        label_tuples = [('label', col) for col in label_cols]
        
        combined_df_multi = combined_df.copy()
        combined_df_multi.columns = pd.MultiIndex.from_tuples(
            feature_tuples + label_tuples
        )
        
        class PrecomputedDataHandler(DataHandler):
            """DataHandler for precomputed data."""
            
            def __init__(self, data_df, segments):
                self._data = data_df
                self._segments = segments
            
            @property
            def data_loader(self):
                return None
            
            @property
            def instruments(self):
                try:
                    return list(self._data.index.get_level_values('instrument').unique())
                except KeyError:
                    return list(self._data.index.get_level_values(1).unique())
            
            def fetch(self, selector=None, level='datetime', col_set='feature',
                     data_key=None, squeeze=False, proc_func=None):
                if col_set in ('feature', 'label'):
                    result = self._data[col_set].copy()
                elif col_set == '__all' or col_set is None:
                    result = self._data.copy()
                else:
                    if isinstance(col_set, (list, tuple)):
                        result = self._data[list(col_set)].copy()
                    else:
                        result = self._data.copy()
                if selector is not None:
                    try:
                        dates = result.index.get_level_values('datetime')
                    except KeyError:
                        dates = result.index.get_level_values(0)
                    if isinstance(selector, tuple) and len(selector) == 2:
                        start, end = selector
                        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                        result = result.loc[mask]
                    elif isinstance(selector, slice):
                        start = selector.start
                        end = selector.stop
                        if start is not None and end is not None:
                            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                            result = result.loc[mask]
                
                if squeeze and result.shape[1] == 1:
                    result = result.iloc[:, 0]
                
                return result
            
            def get_cols(self, col_set='feature'):
                if col_set in self._data.columns.get_level_values(0):
                    return list(self._data[col_set].columns)
                return list(self._data.columns.get_level_values(1))
            
            def setup_data(self, **kwargs):
                pass
            
            def config(self, **kwargs):
                pass
        
        handler = PrecomputedDataHandler(combined_df_multi, dataset_config['segments'])
        dataset = DatasetH(
            handler=handler,
            segments=dataset_config['segments']
        )
        
        logger.debug(f"  Custom factor mode: {len(feature_cols)} factors, {len(combined_df)} rows, train={dataset_config['segments']['train']}")
        
        return dataset
    
    def _compute_label(self, label_expr: str) -> pd.DataFrame:
        """Compute label using Qlib (label requires look-ahead)."""
        from qlib.data import D
        
        data_config = self.config['data']
        
        logger.debug(f"  Label expr: {label_expr}")
        
        stock_list = D.instruments(data_config['market'])
        
        label_df = D.features(
            stock_list,
            [label_expr],
            start_time=data_config['start_time'],
            end_time=data_config['end_time'],
            freq=self.freq
        )
        
        label_df.columns = ['LABEL0']
        
        logger.debug(f"  Label rows: {len(label_df)}")
        
        return label_df
    
    def _load_qlib_factors(self, factor_expressions: Dict[str, str]) -> Optional[pd.DataFrame]:
        """Load Qlib-compatible factors."""
        from qlib.data import D
        
        data_config = self.config['data']
        
        try:
            stock_list = D.instruments(data_config['market'])
            
            expressions = list(factor_expressions.values())
            names = list(factor_expressions.keys())
            
            df = D.features(
                stock_list,
                expressions,
                start_time=data_config['start_time'],
                end_time=data_config['end_time'],
                freq=self.freq
            )
            
            df.columns = names
            return df
        except Exception as e:
            logger.warning(f"Failed to load Qlib factors: {e}")
            return None
    
    def _train_and_backtest(self, dataset, exp_name: str, rec_name: str, output_name: Optional[str] = None) -> Dict:
        """Train model and run backtest."""
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data import D
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
        from qlib.backtest import backtest as qlib_backtest
        from qlib.contrib.evaluate import risk_analysis
        
        model_config = self.config['model']
        backtest_config = self.config['backtest']['backtest']
        strategy_config = self.config['backtest']['strategy']

        # 重定向 mlflow 跟踪目录到可写路径，规避默认 ./mlruns 在部分环境下
        # 因残留 run / 权限导致写 tags/mlflow.runName 抛 PermissionError。
        import mlflow as _mlflow
        _mluri = os.environ.get("MLFLOW_TRACKING_URI") or str(project_root / "data" / "mlruns")
        os.makedirs(_mluri, exist_ok=True)
        try:
            _mlflow.set_tracking_uri(_mluri)
            logger.info("mlflow tracking uri = %s", _mlflow.get_tracking_uri())
        except Exception as _e:  # noqa: BLE001
            logger.warning("设置 mlflow tracking uri 失败（忽略）: %s", _e)

        metrics = {}
        
        with R.start(experiment_name=exp_name, recorder_name=rec_name):
            # Train model
            train_start = time.time()
            
            if model_config['type'] == 'lgb':
                model = LGBModel(**model_config['params'])
            else:
                raise ValueError(f"Unsupported model type: {model_config['type']}")
            
            model.fit(dataset)
            print(f"[4/4] Train LightGBM done ({time.time()-train_start:.1f}s)")
            
            # Generate prediction
            pred = model.predict(dataset)
            logger.debug(f"  Pred shape: {pred.shape}")
            
            # Save prediction
            sr = SignalRecord(recorder=R.get_recorder(), model=model, dataset=dataset)
            sr.generate()
            
            # Compute IC metrics
            try:
                sar = SigAnaRecord(recorder=R.get_recorder(), ana_long_short=False, ann_scaler=self.ann_scaler)
                sar.generate()
                
                recorder = R.get_recorder()
                try:
                    ic_series = recorder.load_object("sig_analysis/ic.pkl")
                    ric_series = recorder.load_object("sig_analysis/ric.pkl")
                    
                    if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
                        metrics['IC'] = float(ic_series.mean())
                        metrics['ICIR'] = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
                    
                    if isinstance(ric_series, pd.Series) and len(ric_series) > 0:
                        metrics['Rank IC'] = float(ric_series.mean())
                        metrics['Rank ICIR'] = float(ric_series.mean() / ric_series.std()) if ric_series.std() > 0 else 0.0
                    
                    print(f"  IC={metrics.get('IC', 0):.6f}, ICIR={metrics.get('ICIR', 0):.6f}, "
                          f"Rank IC={metrics.get('Rank IC', 0):.6f}, Rank ICIR={metrics.get('Rank ICIR', 0):.6f}")
                except Exception as e:
                    logger.warning(f"Could not read IC result: {e}")
            except Exception as e:
                logger.warning(f"IC analysis failed: {e}")
            # Portfolio backtest
            try:
                bt_start = time.time()
                
                market = self.config['data']['market']
                instruments = D.instruments(market)
                stock_list = D.list_instruments(
                    instruments,
                    start_time=backtest_config['start_time'],
                    end_time=backtest_config['end_time'],
                    as_list=True
                )
                logger.debug(f"  Stock count: {len(stock_list)}")
                if len(stock_list) < 10:
                    logger.warning(f"Stock pool too small ({len(stock_list)}), results may be unreliable")
                # Filter invalid price signals
                try:
                    price_data = D.features(
                        stock_list,
                        ['$close'],
                        start_time=backtest_config['start_time'],
                        end_time=backtest_config['end_time'],
                        freq=self.freq
                    )
                    invalid_mask = (price_data['$close'] == 0) | (price_data['$close'].isna())
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        logger.debug(f"  Found {invalid_count} zero/NaN price records")
                        if isinstance(pred, pd.Series):
                            invalid_indices = invalid_mask[invalid_mask].index
                            invalid_set = set()
                            for idx in invalid_indices:
                                instrument, datetime = idx
                                invalid_set.add((datetime, instrument))
                            
                            filtered_count = 0
                            for idx in pred.index:
                                if idx in invalid_set:
                                    pred.loc[idx] = np.nan
                                    filtered_count += 1
                            
                            if filtered_count > 0:
                                logger.debug(f"  Filtered {filtered_count} invalid price signals")
                except Exception as filter_err:
                    logger.warning(f"Price filter failed: {filter_err}")
                
                portfolio_metric_dict, indicator_dict = qlib_backtest(
                    executor={
                        "class": "SimulatorExecutor",
                        "module_path": "qlib.backtest.executor",
                        "kwargs": {
                            # 原来硬编码 "day"：分钟数据集没有 day 日历，
                            # 账户报告会退化到去找 1min 日历并抛
                            # "can't find a freq ... that can resample to 1min"。
                            "time_per_step": self.freq,
                            "generate_portfolio_metrics": True,
                            "verbose": False,
                            "indicator_config": {"show_indicator": False}
                        }
                    },
                    strategy={
                        "class": strategy_config['class'],
                        "module_path": strategy_config['module_path'],
                        "kwargs": {
                            "signal": pred,
                            "topk": strategy_config['kwargs']['topk'],
                            "n_drop": strategy_config['kwargs']['n_drop']
                        }
                    },
                    start_time=backtest_config['start_time'],
                    end_time=backtest_config['end_time'],
                    account=backtest_config['account'],
                    benchmark=backtest_config['benchmark'],
                    exchange_kwargs={
                        "codes": stock_list,
                        **backtest_config['exchange_kwargs']
                    }
                )
                
                print(f"  Portfolio backtest done ({time.time()-bt_start:.1f}s)")
                # Extract portfolio metrics
                # 报告的 key 是回测频率（日频是 "1day"，分钟频是 freq 本身），
                # 原来只认 "1day"，分钟级回测会取不到结果。
                pm_key = None
                if isinstance(portfolio_metric_dict, dict) and portfolio_metric_dict:
                    for candidate in (self.freq, "1day", "5min", "1min"):
                        if candidate in portfolio_metric_dict:
                            pm_key = candidate
                            break
                    if pm_key is None:
                        pm_key = next(iter(portfolio_metric_dict))
                if pm_key:
                    report_df, positions_df = portfolio_metric_dict[pm_key]
                    
                    if isinstance(report_df, pd.DataFrame) and 'return' in report_df.columns:
                        portfolio_return = report_df['return'].replace([np.inf, -np.inf], np.nan).fillna(0)
                        bench_return = report_df['bench'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'bench' in report_df.columns else 0
                        cost = report_df['cost'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'cost' in report_df.columns else 0
                        
                        excess_return_with_cost = portfolio_return - bench_return - cost
                        excess_return_with_cost = excess_return_with_cost.dropna()
                        
                        if len(excess_return_with_cost) > 0:
                            try:
                                daily_df = report_df.copy()
                                daily_df['excess_return'] = excess_return_with_cost
                                
                                output_dir = Path(self.config['experiment'].get('output_dir', './backtest_v2_results'))
                                output_dir.mkdir(parents=True, exist_ok=True)
                                
                                file_prefix = output_name if output_name else exp_name
                                csv_path = output_dir / f"{file_prefix}_cumulative_excess.csv"
                                save_df = daily_df[['excess_return']].copy()
                                save_df.columns = ['daily_excess_return']
                                save_df['cumulative_excess_return'] = save_df['daily_excess_return'].cumsum()
                                
                                save_df.index.name = 'date'
                                save_df.to_csv(csv_path)
                                logger.debug(f"  Daily excess return saved: {csv_path}")
                            except Exception as csv_err:
                                logger.warning(f"Failed to save daily CSV: {csv_err}")

                            # risk_analysis 只认 day/week/month，分钟级必须显式给 N，
                            # 否则会按 252 折算，年化收益/IR 严重失真。
                            analysis = risk_analysis(excess_return_with_cost, N=self.ann_scaler)
                            
                            if isinstance(analysis, pd.DataFrame):
                                analysis = analysis['risk'] if 'risk' in analysis.columns else analysis.iloc[:, 0]
                            
                            ann_ret = float(analysis.get('annualized_return', 0))
                            info_ratio = float(analysis.get('information_ratio', 0))
                            max_dd = float(analysis.get('max_drawdown', 0))
                            
                            if not np.isnan(ann_ret) and not np.isinf(ann_ret):
                                metrics['annualized_return'] = ann_ret
                            if not np.isnan(info_ratio) and not np.isinf(info_ratio):
                                metrics['information_ratio'] = info_ratio
                            if not np.isnan(max_dd) and not np.isinf(max_dd):
                                metrics['max_drawdown'] = max_dd
                            
                            if max_dd != 0 and not np.isnan(ann_ret) and not np.isinf(ann_ret):
                                calmar = ann_ret / abs(max_dd)
                                if not np.isnan(calmar) and not np.isinf(calmar):
                                    metrics['calmar_ratio'] = calmar
                            
            except Exception as e:
                logger.warning(f"Portfolio backtest failed: {e}")
                import traceback
                traceback.print_exc()
        
        return metrics
    
    def _print_results(self, metrics: Dict, total_time: float):
        """Print result summary."""
        def _f(val, fmt='.6f'):
            return format(val, fmt) if isinstance(val, (int, float)) else 'N/A'

        print(f"\n{'='*50}")
        print("Backtest Results")
        print(f"{'='*50}")
        print("[IC Metrics]")
        print(f"  IC: {_f(metrics.get('IC'))}  ICIR: {_f(metrics.get('ICIR'))}")
        print(f"  Rank IC: {_f(metrics.get('Rank IC'))}  Rank ICIR: {_f(metrics.get('Rank ICIR'))}")
        print("[Strategy Metrics]")
        print(f"  Ann. Return: {_f(metrics.get('annualized_return'), '.4f')}  Max DD: {_f(metrics.get('max_drawdown'), '.4f')}")
        print(f"  Info Ratio: {_f(metrics.get('information_ratio'), '.4f')}  Calmar: {_f(metrics.get('calmar_ratio'), '.4f')}")
        print(f"Total time: {total_time:.1f}s")
        print(f"{'='*50}")
    
    def _save_results(self, metrics: Dict, exp_name: str, 
                     factor_source: str, num_factors: int, elapsed: float,
                     output_name: Optional[str] = None):
        """Save results."""
        output_dir = Path(self.config['experiment'].get('output_dir', './backtest_v2_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_name:
            output_file = f"{output_name}_backtest_metrics.json"
        else:
            output_file = self.config['experiment']['output_metrics_file']
        output_path = output_dir / output_file
        
        result_data = {
            "experiment_name": exp_name,
            "factor_source": factor_source,
            "num_factors": num_factors,
            "metrics": metrics,
            "config": {
                "data_range": f"{self.config['data']['start_time']} ~ {self.config['data']['end_time']}",
                "test_range": f"{self.config['dataset']['segments']['test'][0]} ~ {self.config['dataset']['segments']['test'][1]}",
                "backtest_range": f"{self.config['backtest']['backtest']['start_time']} ~ {self.config['backtest']['backtest']['end_time']}",
                "market": self.config['data']['market'],
                "benchmark": self.config['backtest']['backtest']['benchmark']
            },
            "elapsed_seconds": elapsed
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved: {output_path}")
        summary_file = output_dir / "batch_summary.json"
        summary_data = []
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            except:
                summary_data = []
        
        ann_ret = metrics.get('annualized_return')
        mdd = metrics.get('max_drawdown')
        calmar_ratio = None
        if ann_ret is not None and mdd is not None and mdd != 0:
            calmar_ratio = ann_ret / abs(mdd)
        
        summary_entry = {
            "name": output_name or exp_name,
            "num_factors": num_factors,
            "IC": metrics.get('IC'),
            "ICIR": metrics.get('ICIR'),
            "Rank_IC": metrics.get('Rank IC'),
            "Rank_ICIR": metrics.get('Rank ICIR'),
            "annualized_return": ann_ret,
            "information_ratio": metrics.get('information_ratio'),
            "max_drawdown": mdd,
            "calmar_ratio": calmar_ratio,
            "elapsed_seconds": elapsed
        }
        summary_data.append(summary_entry)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Appended to summary: {summary_file}")

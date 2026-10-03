import io
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

# render it with jinja
from jinja2 import Environment, StrictUndefined

from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.utils.env import QTDockerEnv
from quantaalpha.log import logger


def generate_data_folder_from_qlib(use_local: bool = True):
    # ------------------------------------------------------------------
    # 分钟级挖掘支持（2026-10-04 新增）
    #
    # 因子算在什么频率上，只取决于 factor.py 读的 ./daily_pv.h5，
    # 与回测侧 conf 的 provider_uri / freq 是两条独立的线。
    # 设 QUANTAALPHA_FACTOR_FREQ=5min 后，这里会改用 5 分钟版的价量数据，
    # 否则分钟级挖掘会出现「回测用 5min、因子还是日频」的错位
    # （实测后果：17 条轨迹训出同一个基线模型，RankIC 全部相同）。
    #
    # 5min 版由 `python tools/gen_pv_5min.py` 生成。
    # ------------------------------------------------------------------
    factor_freq = os.environ.get("QUANTAALPHA_FACTOR_FREQ", "day").strip().lower()
    if factor_freq not in ("day", "5min"):
        raise ValueError(f"QUANTAALPHA_FACTOR_FREQ 只支持 day/5min，收到 {factor_freq!r}")

    template_path = Path(__file__).parent / "data_template"
    # 5min 版体积 200MB+，不往 data_template 里再拷一份；直接读生成目录
    # （两处都被 .gitignore 覆盖：*.h5 与 git_ignore_folder/）。
    project_root = Path(__file__).parent.parent.parent
    src_5min = Path(
        os.environ.get(
            "QUANTAALPHA_FACTOR_PV_5MIN",
            project_root / "git_ignore_folder" / "factor_implementation_source_data_5min" / "daily_pv.h5",
        )
    )
    daily_pv_all = src_5min if factor_freq == "5min" else template_path / "daily_pv_all.h5"
    daily_pv_debug = daily_pv_all  # 5min 场景不再区分 debug 子集，直接用同一份

    if factor_freq == "5min":
        if not daily_pv_all.exists():
            raise FileNotFoundError(
                f"5min 价量数据缺失: {daily_pv_all}\n"
                f"请先生成：python tools/gen_pv_5min.py"
            )
        # 安全阀：把 217MB 的 5min h5 写进日频 debug 目录是不可逆的破坏
        # （日频挖掘会静默算在分钟数据上，症状极难定位）。必须显式指定 5min 专用目录。
        for name, folder in (
            ("FACTOR_CoSTEER_DATA_FOLDER", FACTOR_COSTEER_SETTINGS.data_folder),
            ("FACTOR_CoSTEER_DATA_FOLDER_DEBUG", FACTOR_COSTEER_SETTINGS.data_folder_debug),
        ):
            if "5min" not in str(folder).lower():
                raise ValueError(
                    f"QUANTAALPHA_FACTOR_FREQ=5min 时 {name} 必须指向 5min 专用目录，"
                    f"当前是 {folder!r}。请设置：\n"
                    f"  {name}=git_ignore_folder/factor_implementation_source_data_5min\n"
                    f"（日频目录必须保持日频数据，不能混用）"
                )
        logger.info(f"Factor source data freq=5min -> {daily_pv_all.name}")
        _copy_to_data_folders(daily_pv_all, template_path)
        return

    # Windows/本机：h5 已预生成时直接跳过全市场 D.features（该步在 Windows 上极慢/卡死）。
    # 只有 h5 缺失时才真正走 qlib 生成（Docker 或 Linux 主机）。
    if daily_pv_all.exists() and daily_pv_debug.exists():
        logger.info("Factor source data (daily_pv_all/debug.h5) already present; skipping qlib generation")
    else:
        qtde = QTDockerEnv(is_local=use_local)
        qtde.prepare()

        # Use current Python so subprocess finds installed packages (e.g. qlib)
        python_exe = sys.executable
        logger.info(f"Generating factor data in {'local' if use_local else 'Docker'}")
        qtde.run(
            local_path=str(template_path),
            entry=f'{python_exe} generate.py',
        )

        # Check that file was generated
        assert daily_pv_all.exists(), "daily_pv_all.h5 is not generated."
        assert daily_pv_debug.exists(), "daily_pv_debug.h5 is not generated."

    # Create data dir and copy files
    logger.info("Copying generated data files to workspace")
    _copy_to_data_folders(daily_pv_all, template_path, debug_src=daily_pv_debug)

    logger.info("Data preparation done")


def _copy_to_data_folders(src_h5: Path, template_path: Path, debug_src: Path | None = None) -> None:
    """把价量 h5 + README 复制进 factor 数据目录（正式目录与 debug 目录）。

    抽出来是因为 5min 分支也要走同一段逻辑，且 5min 用独立的 README
    （必须告诉 LLM 数据频率与窗口含义，否则它会按日频直觉写 TS_MEAN(x,20)）。
    """
    is_5min = src_h5.name.endswith("5min.h5")
    readme_name = "README_5min.md" if is_5min else "README.md"
    readme_src = template_path / readme_name
    if not readme_src.exists():  # 兜底：5min README 缺失时退回通用版
        readme_src = template_path / "README.md"

    for folder, h5_src in (
        (FACTOR_COSTEER_SETTINGS.data_folder, src_h5),
        (FACTOR_COSTEER_SETTINGS.data_folder_debug, debug_src or src_h5),
    ):
        dest = Path(folder)
        dest.mkdir(parents=True, exist_ok=True)
        # h5 200MB+，大小一致就不重复拷（同一轮多次进入这个函数的场景）
        dst_h5 = dest / "daily_pv.h5"
        if not dst_h5.exists() or dst_h5.stat().st_size != h5_src.stat().st_size:
            shutil.copy(h5_src, dst_h5)
        # README 很小，每次覆盖，保证 LLM 拿到的是当前频率的说明
        shutil.copy(readme_src, dest / "README.md")
        logger.info(f"Factor source data -> {dest} ({h5_src.name}, {readme_src.name})")



def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.

    Returns
    -------
    str
        The description of the file.
    """
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string(
        """
{{file_name}}
```{{type_desc}}
{{content}}
```
"""
    )

    if p.name.endswith(".h5"):
        df = pd.read_hdf(p)
        # get df.head() as string with full width
        pd.set_option("display.max_columns", None)  # or 1000
        pd.set_option("display.max_rows", None)  # or 1000
        pd.set_option("display.max_colwidth", None)  # or 199

        if isinstance(df.index, pd.MultiIndex):
            df_info = f"MultiIndex names:, {df.index.names})\n"
        else:
            df_info = f"Index name: {df.index.name}\n"
        columns = df.dtypes.to_dict()
        filtered_columns = [f"{i, j}" for i, j in columns.items() if i in variable_list]
        if filtered_columns:
            df_info += "Related Data columns: \n"
            df_info += ",".join(filtered_columns)
        else:
            df_info += "Data columns: \n"
            df_info += ",".join(columns)
        df_info += "\n"
        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += f"""
A snapshot of one instrument, from which you can tell the distribution of the data:
{df_on_one_instrument.head(5)}
"""
        return JJ_TPL.render(
            file_name=p.name,
            type_desc="h5 info",
            content=df_info,
        )
    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="markdown",
                content=content,
            )
    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def get_data_folder_intro(
    fname_reg: str = ".*",
    flags=0,
    variable_mapping=None,
    use_local: bool = True,
) -> str:
    """
    Directly get the info of the data folder.
    It is for preparing prompting message.

    Parameters
    ----------
    fname_reg : str
        a regular expression to filter the file name.

    flags: str
        flags for re.match

    Returns
    -------
        str
            The description of the data folder.
    """

    # 原判断只看"目录是否存在"，导致目录里已有 h5 但 README 缺失/版本不对时
    # 不会触发同步 —— 实测后果：5min 数据到位了，但 LLM 读到的还是日频 README，
    # 于是因子名写成 10D/20D/5D、窗口按日频直觉取，语义全错。
    # 注意 LLM 看到的是 **debug 目录** 的说明（下面 iterdir 用的是 data_folder_debug）。
    df_main = Path(FACTOR_COSTEER_SETTINGS.data_folder)
    df_debug = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    if (
        not df_main.exists()
        or not df_debug.exists()
        or not (df_main / "daily_pv.h5").exists()
        or not (df_debug / "README.md").exists()
    ):
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_data_folder_from_qlib(use_local=use_local)
    content_l = []
    
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)

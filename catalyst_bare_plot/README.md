# hypotheses_compare_xend_36

配对比较 baseline 实验和策略实验：一组 baseline `<base>` 对应一组策略实验 `<base>_<suffix>`，trial 按名称配对。

不加参数时，生成 `D:\UMich-Smarticles\videos\1105_catalyst_amount_xend\datafile\hypotheses_compare_xend_36` 的全部结果（84 个文件），输入是 `1029_test_catalyst/datafile` 下的 6 个 baseline（a016 a034 a062 a816 a834 a862）和 `1105_catalyst_amount_xend/datafile` 下对应的 `a*_xend_36`。

## 运行

```powershell
conda activate smarticle-3.11
python run_analysis.py                                 # 默认输入，全部步骤
python run_analysis.py --steps bars overview           # 只跑部分步骤

# 换输入数据
python run_analysis.py `
  --baseline-root D:\数据\baseline `
  --strategy-root D:\数据\strategy `
  --suffix xend_12 `
  --bases a016 a034 a062 `
  --out-root D:\结果\xend_12
```

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--baseline-root` | 存放 baseline 实验的文件夹 | `1029_test_catalyst/datafile` |
| `--strategy-root` | 存放策略实验的文件夹 | `1105_catalyst_amount_xend/datafile` |
| `--suffix` | 策略实验名为 `<base>_<suffix>` | `xend_36` |
| `--bases` | baseline 实验名 | 上述 6 个 |
| `--out-root` | 输出文件夹，同名文件会被覆盖 | `<strategy-root>/hypotheses_compare_<suffix>` |
| `--t50-bases` | 额外输出 t ≥ 50 s 窗口的组 | `a834` |
| `--snapshot-bases` | 要画快照的组 | `--bases` 的前 3 个 |
| `--snapshot-trial` | 画快照的 trial | `trial_0002` |
| `--ic-file` | 初始条件 JSON | 策略实验 `config_snapshot.json` 里的 `INIT_FILE` |
| `--ylim` | composition 柱状图的 y 范围，超出部分会被截断 | `-30 30` |
| `--steps` | 要跑的步骤 | 全部 |

默认路径写在 `run_analysis.py` 开头。

## 合并多批数据

同一实验分几批跑（例如 IC 11–15 和 IC 16–20，各自都是 `trial_0000`–`trial_0004`）时，先用 `merge_batches.py` 合成一个实验，再把合并后的文件夹当作 root 传给 `run_analysis.py`：

```powershell
$M = "C:\Users\tianyihu\smarticle_merged\sync_11-20"
$IC = "E:\1106_catalyst_fixed\init_conditions\init_conditions_200_p_N500.json"
python merge_batches.py --out $M\baseline --batches E:\1107_baseline\sync E:\1107_baseline\16-20 `
  --exps a816 a834 a862 --run-tracks --init-file $IC
python merge_batches.py --out $M\strategy --batches E:\1106_catalyst_fixed\datafile E:\1106_catalyst_fixed\datafile_16-20 `
  --exps a816_12-852 a834_12-852 a862_12-852 --run-tracks --init-file $IC
python run_analysis.py --baseline-root $M\baseline --strategy-root $M\strategy --suffix 12-852 `
  --bases a816 a834 a862 --out-root E:\1106_catalyst_fixed\analysis\sync_12_11-20
python batch_split.py E:\1106_catalyst_fixed\analysis\sync_12_11-20 --strategy-root $M\strategy
```

- **原理：** 不复制数据。合并文件夹里的 `trial_0005` 等是指向原 trial 文件夹的 junction，trial 按 `--batches` 的顺序重新编号。`trial_stats`、`tracks`、`lifetime_trials` 三张表合并后改用新 trial 名，配置取第一批，`INIT_INDICES` 拼接起来。`_merge_map.csv` 记录每个新 trial 来自哪一批、原 trial 名、IC 编号和 `RUNTIME_GAIT_SWITCH_MODE`。
- **文件名不改：** junction 里的文件保留原名，例如 `trial_0005` 里是 `trial_0000_groups_d65.csv`。`strategy_hypotheses.py` 和 `render_snapshots.py` 按通配符找文件、从文件夹名取 trial 名，所以能用。`group_lifetimes.py` 按 trial 名拼文件名，所以 `--run-tracks` 会在每批原文件夹里各自运行它，生成它平时也会写的缓存文件。
- **合并文件夹必须放在 NTFS 盘上：** exFAT 不支持 junction，E 盘和 D 盘都是 exFAT，原数据可以在任何盘。
- **删除合并文件夹：** 先用 `rmdir`（cmd）删掉 junction，不要递归删除，否则可能删到原数据。
- **两批配置不同时会提示：** 除 IC、输出路径等预期会不同的键以外，其他配置不同时会打印 NOTE。`ENABLE_RUNTIME_GAIT_CONTROL` 为 `False` 时，`RUNTIME_GAIT_*` 不起作用，不会提示（sync_12_11-20 的两批 `RUNTIME_GAIT_SWITCH_MODE` 不同，就属于这种情况）。
- **`batch_split.py`：** 把每组的配对差值按批拆开，写出 `batch_split_last100.csv`，用来检查两批的结论是否一致。

## 输入数据要求

运行前脚本会检查这些要求，并一次列出所有缺失项：

- 文件夹名是 `<base>` 和 `<base>_<suffix>`，分别放在两个 root 下（两个 root 可以相同）。
- 每个实验有 `config_snapshot.json`，配对的两个实验 trial 名称相同。
- 每个 trial 有 `trial_*_groups_d65.csv`。
- 要画快照的实验有 `<exp>_trial_stats.csv` 和对应 trial。

这些要求不会被检查，要自己确认：

- `render_snapshots.py` 还要读 `POS_ALL.csv`；没有 `*_render.npz` 的 trial 只画身体，不画手臂和外环。
- `ic` 步骤用 `config_snapshot.json` 里的 `INIT_FILE` 和 `INIT_INDICES`。数据搬动过的话，`INIT_FILE` 可能已失效，这时用 `--ic-file` 指定。

groups 缓存由 `batch_group_plots.py` 从 `POS_ALL.csv` 生成。`experiments.py` 给出的命令如下（xend_36 的缓存生成于 2026-11-05，当时实际用的命令没有记录）：

```powershell
python batch_group_plots.py "<out_root>/*" --max-dist 65 --summary groups.csv
```

## 换数据时要注意

- **只有 `tracks` 步骤会写入输入文件夹：** 实验缺少 `<exp>_tracks_d65_m3_w1.csv` 时才运行，已有就跳过。
- **不要混用输出目录：** `overview` 会读取 `--out-root` 下所有组的结果，复用旧目录时，已不在 `--bases` 里的旧组也会被画进去。
- **柱状图可能被截断：** y 范围默认固定为 ±30，效应更大时要调 `--ylim`。
- **这些设置仍是固定的：** 分析窗口（最后 100 s）、分组距离 d65、大群 ≥6、成群 ≥3、快照时间点（0、300、500 s），以及时间序列图的平滑窗口（300 帧，60 fps 下是 5 s）。
- **内存不足会中止：** 可用内存低于 1.5 GB 或磁盘低于 5 GB 时入口脚本会停止，`render_snapshots.py` 自身的下限是内存和磁盘各 2 GB。只需重跑未完成的步骤。

## 步骤和输出

| 步骤 | 脚本 | 输出 | 默认文件数 |
|---|---|---|---|
| `tracks` | `group_lifetimes.py` | 写入各实验目录 | 0 |
| `compare` | `group_lifetimes.compare_experiments` | `<pair>/compare_*_lifetime_d65_m3_w1.csv/.png` | 12 |
| `hypotheses` | `strategy_hypotheses.py` | `<pair>/hypotheses_*_last100*`，`--t50-bases` 另有 `_t50*` | 35 |
| `overview` | `hypotheses_overview.py` | `hypotheses_overview_last100.csv/.png` | 2 |
| `bars` | `composition_delta_bars.py` | `composition_delta_*_last100[_bare].csv/.png/.svg`，每图 3 组 | 12 |
| `snapshots` | `render_snapshots.py` | `snapshots_<trial>/<pair>/` | 21 |
| `ic` | `render_ic.py` | `snapshots_<trial>/IC<k>_initial[_annotated].png` | 2 |

`<pair>` 指 `a016_vs_a016_xend_36` 这样的子文件夹。

只作为依赖、不直接运行的文件：`batch_group_plots.py`、`render_trial.py`、`pos_all_grouping.py`、`pos_all_alignment.py`。

## 这份代码和 repo 的关系

这些脚本是独立副本，优先加载本目录的文件，之后修改 `analysis/` 不会影响这里，反过来也一样。

| 文件 | 来源 | 相对来源的改动 |
|---|---|---|
| `group_lifetimes.py` `render_snapshots.py` `render_ic.py` | `analysis/` 工作区，2026-11-06 | 无（`group_lifetimes.py` 含未提交的 `on_frame` 参数，不影响输出） |
| `pos_all_alignment.py` | repo 根目录 | 无 |
| `pos_all_grouping.py` | git `e258c53` | 无 |
| `batch_group_plots.py` | git `e258c53` | 第 165 行 `sys.path.insert(0, …)` 改为 `insert(1, …)` |
| `render_trial.py` | `analysis/` 工作区 | 第 35 行同样改为 `insert(1, …)` |
| `hypotheses_overview.py` `composition_delta_bars.py` | `analysis/` 工作区 | 新增 `--exp-root`；`hypotheses_overview.py` 的 trial 数改为从数据读取 |
| `strategy_hypotheses.py` | `analysis/` 工作区 | 图注里的 trial 数改为从数据读取 |
| `run_analysis.py` | 由当时的一次性串联脚本整理而成 | 输入路径可配置，增加输入检查；groups 缓存按通配符检查（2026-11-15） |
| `merge_batches.py` `batch_split.py` | 新写，2026-11-15 | 合并多批数据，见上文 |

`batch_group_plots.py` 和 `pos_all_grouping.py` 特意取自提交版本：结果生成之后，repo 里这两个文件又做了速度优化，而这里保留的是生成结果时的版本。

`render_trial.py` 在这里只提供 `box`、`rot`、`hex_rgb`。用它渲染视频，以及 `batch_group_plots.py` 的部分图，仍需要 repo 根目录的 `analysis.py`、`config.py` 等模块。

## 验证（2026-11-06）

两组测试都生成了 84 个文件：CSV、PNG 与原结果逐字节一致，SVG 只有日期和随机 ID 不同。

- 默认输入，输出到其他目录，对照 `hypotheses_compare_xend_36`。
- `--suffix xend_12`，输入通过 junction 放到新路径（`--baseline-root`、`--strategy-root`），对照 `1105_catalyst_amount_xend/datafile/hypotheses_compare`。

## 默认数据的注意事项

- **a062 baseline 来源不一：** `trial_0002` 在 2026-11-03 重跑过，其余 4 个 trial 是 10 月 30 日的。重跑前的 tracks 保存在 `a062/_tracks_before_trial_0002_rerun/`。
- **IC13 图：** 原结果里的两张图是从 `1029_test_catalyst/datafile/hypotheses_compare/snapshots_trial_0002` 复制的，这里改为直接生成，内容相同。
- **时间点标注：** 500 s 会落到最后一个有记录的帧，所以文件名是 `t499s`。

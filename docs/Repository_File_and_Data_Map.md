# VR眼动数据分析平台 - 文件与数据全景指南

> 本文档系统梳理 `vr-eyetracking-analysis` 仓库中的代码、脚本、前后端模块、数据集与文档资源，并标注它们之间的依赖关系和典型使用场景，方便快速理解整体工程结构。

## 目录
1. [项目概览](#项目概览)
2. [根目录文档与启动脚本](#根目录文档与启动脚本)
3. [配置与公共资源](#配置与公共资源)
4. [数据资产与派生结果](#数据资产与派生结果)
5. [分析与处理代码模块](#分析与处理代码模块)
6. [可视化与后端服务](#可视化与后端服务)
7. [前端应用](#前端应用)
8. [模型与机器学习支撑](#模型与机器学习支撑)
9. [工具脚本与测试资源](#工具脚本与测试资源)
10. [历史备份与临时文件](#历史备份与临时文件)
11. [核心数据流程与模块关联](#核心数据流程与模块关联)

---

## 项目概览
- **README.md**：提供平台的核心功能、体系架构、技术栈以及使用指南，是进入项目的首要文档，并描述了各大模块（数据处理、RQA分析、模块7/8集成、Web界面等）的职责分工。【README.md†L1-L128】
- **PROJECT_STRUCTURE.md**：以目录树的方式总结数据目录、核心模块、可视化层、配置体系与工具脚本，并给出了主要API端点与模板路径，是本指南的基础参考文档。【PROJECT_STRUCTURE.md†L1-L140】
- **VR眼球追踪数据可视化平台开发文档.md**、**Module10_Complete_Documentation.md** 等中文说明：覆盖模块7~10的开发背景、需求、实现过程和修复记录，为理解项目演化提供上下文。
- `.git`、`.gitignore`：标准Git仓库元数据；`.claude/agents/phd-research-assistant.md` 与 `.claude/settings.local.json` 为协作型AI代理的本地配置，可忽略业务逻辑。

---

## 根目录文档与启动脚本
### 2.1 文档矩阵
| 文件 | 说明 |
| --- | --- |
| `API_DOCUMENTATION.md` | 整理主要后端API接口、参数说明和返回格式。 |
| `DEVELOPMENT_GUIDE.md`、`DEVELOPMENT_LOG.md` | 记录开发约定、进度和关键变更。 |
| `MODULE_SPLIT_SUMMARY.md`、`Module10B_Training_Core.md` 等 | 模块化拆分与模块10的训练/评估设计文档。 |
| `DATA_NORMALIZATION_SUMMARY.md`、`DATA_STRUCTURE_DOCUMENTATION.md` | 描述数据标准化策略、文件结构、字段含义。 |
| 多份中文修复报告（如 `模块8完整修复报告.md`、`数据架构优化报告.md` 等） | 针对特定问题的修复说明和验证结论。 |
| `docs/Module9.1_Data_Preprocessing_Documentation.md` | 已存在的详细预处理说明，与本指南互补。 |

### 2.2 启动与可视化脚本
- **start_server.py**：一键启动 Flask 可视化服务，自动打开浏览器并挂载增强版可视化器。【start_server.py†L1-L52】
- **visualize.py**：命令行入口，支持配置自定义端口、调试模式以及是否自动打开浏览器，调用 `visualization.enhanced_web_visualizer.EnhancedWebVisualizer`。【visualize.py†L1-L79】
- **calibrate.py**：校准主入口，提供控制/MCI/AD分组过滤、题目过滤、就地覆盖、手动偏移以及可视化模式等丰富的命令行参数，内部委托 `calibration` 子模块。【calibrate.py†L1-L120】
- Windows批处理脚本 `一键启动全部服务.bat`、`启动服务器.bat` 用于本地环境的快速执行。

### 2.3 根目录HTML与调试页
- **debug_module10.html**、**test_frontend_params.html**：用于排查模块10导航、API和RQA参数签名的静态测试页，可在浏览器中直接运行 JS 检查逻辑。【debug_module10.html†L1-L34】【test_frontend_params.html†L1-L44】
- **test_frontend_params.html** 同时提供与 `/api/rqa-pipeline/*` 交互的简单前端代码段，便于在无React环境时验证后端接口。

---

## 配置与公共资源
- **config/config.py**：全局配置中心，定义视场角、速度阈值、数据目录映射、任务文件、验证逻辑、统计选项等，并提供 `validate_config()` 与 `show_config_summary()` 辅助函数。【config/config.py†L1-L120】
- **config/calibration_config.json**、**config/eyetracking_analysis_config.json**：分别为校准与眼动分析提供参数、数据源路径、ROI定义、可视化样式等配置，供 `calibration` 和 `analysis/visualization` 模块共享。
- **requirements.txt**：列出Python依赖，覆盖 Flask、NumPy、Pandas、SciPy、Matplotlib、scikit-learn 等核心库。
- **feature_extraction_test_report.html** 等静态报告：展示特征提取或模块调试的结果快照。

---

## 数据资产与派生结果
```
data/
├── ad_raw / ad_processed / ad_calibrated
├── mci_raw / mci_processed / mci_calibrated
├── control_raw / control_processed / control_calibrated
├── event_analysis_results
├── background_images
├── MMSE_Score
├── feature_extraction_results
├── normalized_features
├── module7_integrated_results
├── module8_analysis_results
├── module9_ml_results
├── module10_datasets
├── rqa_results
└── rqa_pipeline_results
```

- **原始与校准数据**：`*_raw`（TXT原始）、`*_processed`（预处理CSV）、`*_calibrated`（应用 `calibration` 偏移后的CSV）。文件命名遵循 `{group}{id}q{task}_preprocessed_calibrated.csv`，具体字段见 `DATA_STRUCTURE_DOCUMENTATION.md`。【DATA_STRUCTURE_DOCUMENTATION.md†L37-L92】
- **event_analysis_results**：包含 `All_Events.csv`、`All_ROI_Summary.csv` 及按组拆分文件，供事件分析与ROI统计模块使用。【DATA_STRUCTURE_DOCUMENTATION.md†L93-L125】
- **background_images/Q1.jpg ~ Q5.jpg**：React前端和Flask可视化用于绘制ROI轮廓与背景图。【PROJECT_STRUCTURE.md†L6-L18】
- **MMSE_Score**：三组MMSE量表CSV及转换公式，用于模块8认知对比。【DATA_STRUCTURE_DOCUMENTATION.md†L9-L36】
- **feature_extraction_results**：保存特征提取阶段生成的中间结果（如 `All_Subjects_Features.csv`）。
- **normalized_features**：包含 `subjects.csv`、`tasks.csv`、`roi_features.csv`、`normalized_features_summary.csv` 等结构化表，记录归一化后的特征矩阵及其数据仓库设计。【data/normalized_features/DATABASE_DESIGN.md†L1-L120】
- **module7_integrated_results**：以RQA参数签名组织的数据集整合摘要与 `metadata.json`，为后续机器学习和可视化提供输入。【data/module7_integrated_results/m2_tau1_eps0.055_lmin2/metadata.json†L1-L80】
- **module8_analysis_results**：每个参数签名下包含眼动系数与MMSE对比的统计结果、导出报表以及 `README.md` 引导使用。【data/module8_analysis_results/README.md†L1-L88】
- **module9_ml_results**：保存特征方向配置、模型评估输出、预测对齐结果等，支撑模块9机器学习服务。
- **module10_datasets**：`m*_tau*_eps*_lmin*` 目录下存放 `Q1.npz` ~ `Q5.npz` 训练数据与 `meta.json` 描述，供模块10训练/服务读取。【data/module10_datasets/m2_tau1_eps0.055_lmin2/meta.json†L1-L80】
- **rqa_results** 与 **rqa_pipeline_results**：前者保存历史递归图和指标，后者按照五步流程（计算、合并、特征补充、统计、可视化）生成结果与图表，在 `analysis/rqa_batch_renderer.py` 与 `visualization/rqa_pipeline_api.py` 中消费。【analysis/rqa_batch_renderer.py†L1-L120】【PROJECT_STRUCTURE.md†L26-L60】
- **analysis_results/data_range_analysis.json**：由 `analysis/data_range_analyzer.py` 生成的统计摘要和归一化配置，为特征标准化提供参考。【analysis/data_range_analyzer.py†L1-L120】

---

## 分析与处理代码模块
### 5.1 `analysis/`
| 文件 | 功能要点 |
| --- | --- |
| `eyetracking_analyzer.py` | 提供速度计算、IVT事件划分、ROI标记、统计聚合等核心分析流程，依赖 `config/eyetracking_analysis_config.json` 中的ROI定义。【analysis/eyetracking_analyzer.py†L1-L160】 |
| `enhanced_eyetracking_analyzer.py` | 在基础分析器上扩展调试输出、数据预处理、事件提取等功能，与增强版可视化器直接集成。【analysis/enhanced_eyetracking_analyzer.py†L1-L120】 |
| `event_analyzer.py` | 包含详尽的ROI硬编码、IVT实现与事件统计函数，是事件分析API与模块8可视化的后端。【analysis/event_analyzer.py†L1-L160】 |
| `rqa_analyzer.py` | 封装RQA指标的计算、批量处理和统计分析，为RQA流程API提供服务。 |
| `rqa_batch_renderer.py` | 负责按参数组合批量渲染递归图、统计指标、生成图像和JSON，并维护结果目录结构。【analysis/rqa_batch_renderer.py†L1-L120】 |
| `data_range_analyzer.py` | 统计游戏时长、ROI注视时间、RQA指标范围并导出归一化配置文件，用于特征标准化策略。【analysis/data_range_analyzer.py†L1-L200】 |
| `analysis_results/` | 默认保存 `data_range_analysis.json`，供数据标准化和可视化摘要使用。 |

### 5.2 `data_processing/`
- **vr_eyetracking_processor.py**：解析TXT原始数据、计算时间差和视角度、滤除异常值，输出预处理CSV。【data_processing/vr_eyetracking_processor.py†L1-L120】
- **custom_vr_parser.py**：兼容不同原始格式的解析器（若干解析规则与异常处理逻辑）。
- **extract_mci_data.py / extract_ad_data.py**：面向MCI和AD组的专用提取脚本，封装特定路径和命名规范。
- **feature_normalizer.py / feature_normalizer_fixed.py**：针对特征归一化流程的实现与修复版本。

### 5.3 `calibration/`
| 文件 | 描述 |
| --- | --- |
| `basic_calibrator.py` | 根据配置对坐标进行线性偏移矫正。 |
| `advanced_calibrator.py` | 支持组级/个体级参数、ROI对齐和批量校准，读取 `calibration_config.json`。 |
| `inplace_calibrator.py` | 直接覆盖原文件的校准方式，可按题号或目录过滤，供 `calibrate.py --inplace` 使用。 |

### 5.4 其他分析支持
- **feature_extraction_test_report.md / .html**：记录特征提取算法的测试输出。
- **analysis/enhanced_eyetracking_analyzer.py** 与 **visualization/enhanced_web_visualizer.py** 双向调用：前者返回统计结果，后者将其渲染为前端图像。

---

## 可视化与后端服务
### 6.1 Flask可视化框架 `visualization/`
| 模块 | 作用 |
| --- | --- |
| `enhanced_web_visualizer.py` | 创建 Flask 应用、注册多种扩展API（RQA、事件、MMSE、真实数据整合、模块9预测、RQA流程），加载背景图和MMSE分数，并暴露 `run_server()`。【visualization/enhanced_web_visualizer.py†L1-L120】 |
| `event_api_extension.py` | 提供事件统计相关的REST接口，与 `analysis/event_analyzer.py` 交互。 |
| `feature_extraction_api.py` | 管理特征提取任务、检查数据源和触发后台处理。 |
| `ml_prediction_api.py` | 对接模块9机器学习预测服务。 |
| `mmse_api_extension.py` | 提供MMSE数据对比接口。 |
| `rqa_api_extension.py` | 调用 `analysis/rqa_batch_renderer.py` 进行批量渲染，管理任务状态。 |
| `rqa_pipeline_api.py` | 实现参数化RQA五步流程的API，与 `data/rqa_pipeline_results` 目录同步。 |
| `real_data_integration_api.py` | 模块7数据整合的接口实现。 |
| `module10_eye_index/` | 模块10相关模板与静态资源。 |
| `templates/`、`static/` | 存放增强版前端页面模板 `enhanced_index.html` 及静态资源。

#### 6.1.1 模板页面结构与交互逻辑
- **`templates/index.html`**：Bootstrap 风格的轻量级可视化入口，包含“研究组概览—数据列表—图像预览—ROI 统计”四区布局，配合 `/api/groups`、`/api/group/<group>/data`、`/api/visualize/<group>/<data>` 与 `/api/statistics/<group>` 接口驱动动态渲染，并内置下载/刷新按钮与 ROI 表格展示。【F:visualization/templates/index.html†L1-L595】
- **`templates/enhanced_index.html`**：整合全部七大可视化子模块的单页壳层：
  - 顶部横幅集成服务器重启、语言切换与数据总量指示，左侧为折叠侧边栏路由到“数据可视化、数据导入、RQA 分析、事件分析、RQA 流程、综合特征、数据整理、智能分析、机器学习、Eye-Index”等视图。【F:visualization/templates/enhanced_index.html†L3563-L3659】【F:visualization/templates/enhanced_index.html†L8045-L8203】
  - 主内容区域为多个占位容器（`visualizationModuleContainer` 等），默认展示加载占位符，并通过 `simple-module-loader.js` 将各模块 HTML 片段注入，再以 `moduleLoaded` 事件触发相应初始化逻辑（如 `initVisualization`、`initDataImport` 等）。【F:visualization/templates/enhanced_index.html†L3668-L3766】【F:visualization/static/js/simple-module-loader.js†L7-L188】【F:visualization/templates/enhanced_index.html†L7278-L7320】
  - 可视化视图本身维护多语言文案、筛选状态与交互控件：`languageTexts` 字典驱动中英文本地化，`toggleLanguage`/`updateLanguage` 同步 UI；页面初始化后批量拉取组别、数据列表、MMSE 量表及统计，并基于当前组/题号过滤列表、调用 `/api/visualize` 生成图像及 `/api/data` 删除或 `/api/mmse-scores` 获取评估详情。【F:visualization/templates/enhanced_index.html†L5840-L6192】【F:visualization/templates/enhanced_index.html†L7529-L7691】【F:visualization/templates/enhanced_index.html†L7701-L7719】【F:visualization/templates/enhanced_index.html†L10448-L10536】
  - 数据整理（模块7）等扩展界面直接嵌入该模板，保留统计卡片、图表容器及操作面板，后续由动态脚本填充内容，便于集中演示归一化特征、AI 辅助分析与模块10-C 性能面板。【F:visualization/templates/enhanced_index.html†L3749-L5827】

### 6.2 模块10后端 `backend/`
```
backend/
├── m10_data_prep
├── m10_training
├── m10_evaluation
├── m10_service
├── expert_demo
├── test_data_api.py
└── test_npz_data.py
```
- **m10_data_prep**：
  - `builder.py`：构造模块10训练所需的特征矩阵与标签。
  - `schema.py`：定义NPZ数据结构与字段含义。
  - `settings.py`：参数签名、默认路径等配置。
- **m10_training**：
  - `dataset.py`、`model.py`、`trainer.py`、`callbacks.py`：PyTorch训练流程；`config.yaml` 存储超参数；`run_train.py` 为训练入口；`utils/logger.py`、`utils/metrics.py` 辅助记录与评估；`api.py` 暴露训练管理接口。【backend/m10_training/dataset.py†L1-L160】【backend/m10_training/trainer.py†L1-L120】
- **m10_evaluation**：
  - `api.py`、`evaluator.py`、`config.py`：模型评估API、评估逻辑与参数配置，使用 `DEVELOPMENT_LOG.md` 追踪迭代。
- **m10_service**：
  - `config.py`：模型根目录、缓存策略、有效任务列表等配置；
  - `versions.py`：版本与模型签名管理；
  - `loader.py` (`ModelManager`)：线程安全的模型加载、激活、缓存与推理。【backend/m10_service/loader.py†L1-L120】
  - `data_api.py`：对外提供NPZ数据表格查询、导出接口。【backend/m10_service/data_api.py†L1-L120】
  - `data_table.py`：NPZ转DataFrame、分页、导出CSV/Excel的工具函数。
  - `metrics.py`：模型指标查询、聚合与缓存逻辑。
  - `predict.py`：预测API蓝图，使用 `ModelManager` 对外服务。
  - `loader.py`、`predict.py`、`metrics.py` 被可视化层 `/api/m10/*` 接口调用。
- **expert_demo**：演示数据与README，用于专家演示环境。
- **测试脚本**：`test_data_api.py`、`test_npz_data.py` 手动验证接口与数据加载；`test_m10c_service.py` 通过HTTP请求全量测试模型服务。【test_m10c_service.py†L1-L120】

### 6.3 其他后端脚本
- **start_server.py** / **visualize.py**：见上节。
- **reload_models.py**：刷新模型缓存或重新加载模块10模型。
- **fix_data_compatibility.py**、`create_better_test_models.py`：辅助脚本，解决历史兼容性或生成更好的测试模型权重。

---

## 前端应用
目录：`frontend/`

- **package.json / package-lock.json**：React前端依赖，`node_modules/` 为第三方库（无需逐一列举）。
- **src/App.js**：定义React Router路由，包含数据可视化、导入、RQA、事件、RQA流程、特征提取、数据整理七大视图。【frontend/src/App.js†L1-L34】
- **src/index.js / index.css / App.css**：入口渲染与全局样式。
- **components/Layout/**：`Layout.js`、`Header.js`、`Sidebar.js` 负责整体布局与导航，CSS文件定义主题样式。
- **components/Common/LanguageSwitch**：提供中英文切换组件。
- **components/Modules/**：
  - `DataVisualization/`：实现数据列表、过滤器、参数控制、可视化展示，与 `apiService.generateVisualization`、全局 `useAppStore` 交互。【frontend/src/components/Modules/DataVisualization/DataVisualization.js†L1-L120】
  - `DataImport/`、`DataOrganization/`、`FeatureExtraction/`、`RQAAnalysis/`、`EventAnalysis/`、`RQAPipeline/`：各模块对应的占位或功能组件，随着后端API完善逐步实现。
- **services/api.js**：封装axios实例，集中管理所有REST调用，包括数据管理、可视化、RQA批处理、RQA流程、事件分析、特征提取等接口。【frontend/src/services/api.js†L1-L120】
- **store/useAppStore.js**：Zustand 状态管理，维护当前视图、语言、数据过滤、加载状态以及通用操作方法。【frontend/src/store/useAppStore.js†L1-L80】
- **utils/languages.js**：双语文案字典，供组件按当前语言选择文本。【frontend/src/utils/languages.js†L1-L48】
- **public/**：React构建所需静态资源与 `index.html` 模板。
- **启动前端.bat**：Windows环境下的启动脚本。

前端通过 API 服务层与 Flask 后端交互，核心可视化模块调用 `/api/visualize/*`、`/api/rqa-*`、`/api/event-analysis/*` 等路由。

---

## 模型与机器学习支撑
- **models/**：按RQA参数签名划分的模型目录（如 `m2_tau1_eps0.055_lmin2`），每个任务 `Q1`~`Q5` 存在 `*_best.pt`、时间戳版本、TorchScript `*.ts` 以及 `*_best_metrics.json`，供 `ModelManager` 列举和加载。【models/m2_tau1_eps0.055_lmin2/Q1_best_metrics.json†L1-L40】
- **create_test_models.py**：生成模拟MLP模型、TorchScript版本及指标文件以测试模块10-C服务，并创建TensorBoard日志结构。【create_test_models.py†L1-L120】
- **create_better_test_models.py**：改进版测试模型生成器，生成更贴近真实性能的权重和指标。
- **reload_models.py**：刷新/重新加载模型缓存，确保模型目录更新后服务可立即使用。
- **backend/m10_training** 与 **backend/m10_service**（见上节）共同构成模型训练、评估、部署的完整闭环。

---

## 工具脚本与测试资源
- **utils/**：
  - `optimize_directory_naming.py`：批量调整目录命名以保持一致性。
  - `process_internal_group1.py`：针对内部数据组的批处理脚本。
  - `test_single_file.py`：在命令行快速验证单个文件的处理流程。
  - `run.bat`：Windows一键执行常用脚本。
- **analysis/analysis_results/data_range_analysis.json**：测试归一化和数据范围分析的产物。
- **test_m10c_service.py**：详尽的HTTP测试脚本，覆盖健康检查、模型列表、激活、单条与批量预测、指标查询、日记上传等场景。【test_m10c_service.py†L1-L120】
- **debug_predict.py**、`test_data_api.py`、`test_npz_data.py`：对预测接口、数据API及NPZ文件进行独立调试的脚本。
- **feature_extraction_test_report.md/html**：验证特征提取模块的输出格式与质量。

---

## 历史备份与临时文件
- **backup_old_control_group_1/**、`backup_old_control_group_2/`：保存历史版本的控制组数据（原始TXT、OCR结果、帧级ROI等），便于追踪数据修复前后的差异。
- **analysis/__pycache__/**、`visualization/__pycache__/` 等：Python编译缓存文件夹，可忽略。
- **runs/**（如存在）：TensorBoard训练日志，由测试模型脚本生成。

---

## 核心数据流程与模块关联
1. **原始采集**：`data/{group}_raw` 中的TXT文件通过 `data_processing/vr_eyetracking_processor.py` 解析与预处理，输出 `*_processed` CSV。
2. **校准阶段**：`calibration/basic_calibrator.py` 或 `advanced_calibrator.py` 根据 `config/calibration_config.json` 计算偏移量，生成 `*_calibrated` CSV，`calibrate.py` 提供批量入口。【calibrate.py†L1-L120】
3. **事件与特征**：校准数据进入 `analysis/eyetracking_analyzer.py`、`analysis/event_analyzer.py`，得到ROI统计、注视/扫视事件、归一化信息，并写入 `event_analysis_results`、`analysis_results` 等目录。
4. **RQA分析**：`analysis/rqa_batch_renderer.py` 或 `visualization/rqa_pipeline_api.py` 根据参数签名运行递归量化分析，产出递归图、统计指标和 `rqa_pipeline_results` 中的多步结果，供前端RQA模块与模块7/8/9调用。【analysis/rqa_batch_renderer.py†L1-L120】
5. **数据整合 (模块7)**：`data/module7_integrated_results` 保存整合后的特征摘要，为机器学习训练和MMSE对比提供统一输入。
6. **机器学习 (模块9/10)**：`backend/m10_training` 读取 `module10_datasets` 下的NPZ数据训练模型，指标保存在 `models/` 与 `module9_ml_results`；`backend/m10_service` 提供模型上线与预测接口。【backend/m10_service/data_api.py†L1-L120】【backend/m10_service/loader.py†L1-L120】
7. **可视化与前端**：`visualization/enhanced_web_visualizer.py` 注册综合API，React前端通过 `frontend/src/services/api.js` 调用，实现数据浏览、RQA流程、事件分析、模块9/10交互等界面功能。【frontend/src/services/api.js†L1-L120】【frontend/src/components/Modules/DataVisualization/DataVisualization.js†L1-L120】

---

## 附录：快速查找索引
- **主要配置**：`config/`、`requirements.txt`
- **后端入口**：`start_server.py`、`visualize.py`
- **数据处理**：`data_processing/`、`calibration/`
- **分析与可视化**：`analysis/`、`visualization/`
- **前端**：`frontend/`
- **模型相关**：`models/`、`backend/m10_*`
- **数据成果**：`data/normalized_features`、`data/module7_*`、`data/module8_*`、`data/module9_*`、`data/module10_*`

> 如需对特定文件或模块深入学习，可结合本文档的路径索引与原始源码/数据进行交叉参考。

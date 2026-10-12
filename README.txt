二进制多层语义研究项目统一代码交付版
==================================

版本基准：2026-10-12；漏洞检测主线：v0_3_8。
范围：研究内容一 统一语义表示；研究内容二 联合表示与跨层推理；
研究内容三的语义恢复和漏洞检测。2.3.3 恶意软件行为识别与家族归因
尚未发现独立实现，本仓库不将其标为已完成。

components/ 是唯一物理源码目录，按功能组织。
packages/ 只包含相对符号链接，用于保留历史绝对路径的兼容性，不是源码副本。
文件名中的版本号表示算法来源或冻结协议，不代表提供多份整项目历史版本。
release/source_manifest.json 逐文件记录原服务器路径、章节和 SHA-256。
release/code_inventory.json 提供全部文件及 Python 顶层符号、行号和导入列表。
docs/项目代码总体说明.docx 是按总说明书章节组织的阅读文档。

本次更新是源码选版、结构整理、依赖核查与文档化，不修改算法逻辑，
不重新训练模型、不重新执行漏洞实验、不改写实验结果。
“统一交付版”不等于说明书全部算法已实现或全部实验已通过。

功能组件
--------
research_pipeline     2.1 和 2.2 的 BlockSem 上游与表示推理流水线
usg_compat            现有 SERAPH 和 v038 数据契约仍依赖的 USG 提取器
semantic_export       伪代码、指令和语义查询的导出接口
llm_adaptation        第一阶段 clean LoRA 适配
symbolic_encoder      clean 符号编码器训练和向量导出
semantic_joint        clean 第二阶段联合排序模型
data_split            项目级全局划分与候选规范化
instruction_mapping   反编译伪代码与指令的精确映射
gold_mapping          独立 Gold 映射工具
test_data             FINAL_TEST_V2 数据准备和隔离
semantic_freeze       最终冻结契约校验
semantic_evaluation   FINAL_TEST_V2 预测与独立评测
ssq_transport         可选 SSQ 审计的 API 传输依赖
ssq_design            可选 SSQ 审计的固定设计协议依赖
ssq_audit             SSQ 候选顺序与平分诊断，不作为默认增强
security_semantics    安全语义适配、过程间数据流和风险模型能力模块
vulnerability         v038 冻结规则检测和定位实验，保留 v037 数据构建依赖
localization_evaluation 独立定位指标评测
data_tools            共享样本清单、语义输出统计和独立 DWARF 评测工具

部署和数据
----------
统一源码目录拟部署至 /data/chaoni/mjy/binary。
既有数据、模型、Ghidra、JDK 和结果仍位于 /data/chaoni/mjy/resym_exp
及 /data/chaoni/mjy/DeepSeek-Coder-6.7B-Base 等原路径；它们不入 Git。
算法内保留原实验绝对路径和冻结哈希，所以这不是脱离数据即可一键运行的包。
不要运行所有脚本。应按 Word 中的任务入口选择单一流水线。
不要以“v037 是历史版本”为由删除 training_runs/seraph_task2/v0_3_7：
v038 明确依赖该目录中的冻结样本和清单。

清理规则
--------
必须等待 v038 实验结束并核对源文件未改变，再安装旧路径兼容链接。
仅按经核实的清单迁移历史源代码；数据、模型、训练结果、封存 Gold 和
Ghidra 运行目录禁止按版本号清理。历史代码先移出活动目录并保留可恢复归档。
当前部署与清理事实请以 release/ 中的状态记录及交付消息为准。

验证边界
--------
tools/verify_release.py 是只读验证工具：检查源码哈希、Python 语法、
兼容链接及被直接引用的旧包名称。它不会导入研究代码，不会训练、分析目标
或发起网络请求。第三方依赖和 GPU 实验需要原服务器环境。
Python 3.10 及以上用于纯 Python 组件；Ghidra 导出脚本由其 Jython 环境执行。
核心运行依赖包括 PyTorch、transformers、peft、numpy、pyelftools 和 z3-solver，
具体版本应沿用已训练模型的原环境，不能盲目升级。

重要实现边界
------------
research_pipeline 中有 Stage5 对齐、约束处理、迭代回退和四类语义推断。
semantic_joint 当前配置冻结 LLM 和 Stage1 adapter，关闭 constraint_loss；
不要把其他模块的能力当作当前训练实际启用的能力。
四个 infer 文件调用共享的 module_specialized_v6_6 评分器。
security_semantics 的 GNN/过程间能力不是 v038 默认执行链自动包含的阶段。
v038 的效果由独立运行产物决定，源码整理本身不产生新的实验结论。

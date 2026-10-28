# Mini Qwen VL

> 基于 Qwen2.5-VL 的多模态小模型实践：**图表指令微调（SFT）+ 多模态检索增强生成（mRAG）**。

![base](https://img.shields.io/badge/base-Qwen2.5--VL--7B-6f42c1) ![train](https://img.shields.io/badge/train-TRL%20SFT%20%2B%20LoRA-orange) ![rag](https://img.shields.io/badge/rag-ColQwen2%20%2B%20Reranker-0ea5e9)

---

## 项目概览

这个项目做了两件互相独立、但可以串联的事：

| 任务 | 目标 | 入口脚本 |
|---|---|---|
| **SFT 微调** | 让模型看得懂图表，能回答 ChartQA 类型的视觉问答 | `qwen25vl_sft.py` |
| **mRAG 评估** | 从 PDF 文档中检索相关页面，再让模型基于检索结果回答 | `qwen25vl_mRAG_eval.py` |
| **mRAG 数据合成** | 自动从 PDF 生成高质量多模态 QA 对 | `qwen25vl_mRAG_eval_data.py` |

两者共享同一套图像处理与推理工具（`utils/`）。

---

## 目录结构

```
mini-qwen-vl/
├── qwen25vl_sft.py               # ⭐ LoRA 微调（TRL SFTTrainer + DeepSpeed）
├── qwen25vl_eval.py              # SFT 后模型的评测
├── qwen25vl_mRAG.py              # mRAG 推理主流程
├── qwen25vl_mRAG_eval.py         # mRAG 效果评估
├── qwen25vl_mRAG_eval_data.py    # 从 PDF 合成 QA 评测数据
├── accelerate_config.yaml        # 4 卡 DeepSpeed 配置
├── download_data.sh              # 下载 ChartQA / pdfvqa
├── download_models.sh            # 下载 Qwen2.5-VL 与检索/重排模型
├── requirements.txt
└── utils/
    ├── utils.py                  # 图像/PDF 处理、VLM 批量生成
    ├── pdf2img.py                # PDF → 页面图片
    └── templates.py              # QA 生成与自检提示词模板
```

---

## 环境准备

```bash
pip install -r requirements.txt
```

> 注意：`torch==2.5.1+cu118` 需要 CUDA 11.8 环境；`deepspeed` 仅多卡训练时必需。

下载数据与模型：

```bash
bash download_data.sh      # ChartQA → data/sft, pdfvqa → data/mrag
bash download_models.sh    # Qwen2.5-VL-7B + ColQwen2 + MonoQwen2-VL
```

> 脚本里的绝对路径（`/archive/share/...`）是原始集群路径，**运行前请改成你自己的路径**。

---

## 一、SFT 微调

采用 **LoRA + bf16 + DeepSpeed**，在 ChartQA 上做指令微调。

```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 python qwen25vl_sft.py

# 多卡（4 卡，DeepSpeed ZeRO）
accelerate launch --config_file accelerate_config.yaml qwen25vl_sft.py
```

关键配置：

| 项 | 值 |
|---|---|
| 微调方式 | LoRA（`r=4`, `alpha=16`, `dropout=0.05`） |
| 精度 | bf16 |
| 梯度累积 | 16（见 `accelerate_config.yaml`） |
| 数据切分 | train / val / test ≈ 81% / 9% / 10%，固定 `seed=42` |
| 样本格式化 | `utils.format_data_chartqa` |

---

## 二、mRAG：多模态检索增强

传统 RAG 检索的是文本块，而这里检索的是**页面图像**——因为图表、公式、表格在 PDF 转文本时会大量失真。

```
PDF ──pdf2image──▶ 页面图片
                      │
                      ▼
              ColQwen2 向量索引 (Byaldi)
                      │
   用户提问 ──────────▶ 检索 Top-K 页面
                      │
                      ▼
              Reranker 精排 (ColBERT / MonoQwen2-VL)
                      │
                      ▼
        Qwen2.5-VL 读图作答 ◀── 检索页面 + 问题
```

```bash
CUDA_VISIBLE_DEVICES=0 python qwen25vl_mRAG.py
```

---

## 三、QA 数据合成（自举训练数据）

`qwen25vl_mRAG_eval_data.py` 用「生成 → 批判 → 重写」三步循环自动产出高质量 QA 对：

1. 用 VLM 看页面图片**生成问题与答案**；
2. 用另一轮提示让模型**自我批判**（`question_groundedness_critique_prompt` / `question_standalone_critique_prompt`），检查问题是否真的能被该页内容回答、是否脱离上下文后仍可独立理解；
3. 不合格的重新生成。

```bash
export DEEPSEEK_API_KEY="your_key"    # 需要 LLM 参与生成与批判
CUDA_VISIBLE_DEVICES=0 python qwen25vl_mRAG_eval_data.py
```

产出：`logs/eval.json`。

---

## 踩坑记录

- **路径硬编码**：所有脚本顶部都有 `MODEL_PATH` / `DATA_PATH` / `LOG_PATH` 常量，迁移环境时优先改这里。
- **显存**：7B + bf16 + LoRA 单卡约需 24GB；批量推理时注意 `vlm_generate_multi` 的并发数。
- **PDF 转图分辨率**：`pdf2img.py` 的 DPI 直接影响检索与阅读效果，图表密集的文档建议 ≥ 150 DPI。
- **检索与生成解耦**：换检索模型不需要重训 VLM，但需要重建索引。

---

## 参考

- Qwen2.5-VL: <https://github.com/QwenLM/Qwen2.5-VL>
- ColQwen2 / ColPali: 页面级视觉检索
- Byaldi: 多模态索引库
- ChartQA / pdfvqa: 评测数据集

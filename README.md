# Answer Helper - 实时屏幕 OCR 答题助手

Windows 桌面小工具：框选屏幕上的题目区域，本地 OCR 离线识别，优先在题库中模糊匹配，未命中时自动交给大模型解答，答案显示在置顶小悬浮窗中。

## 功能特性

- **离线 OCR**：基于 [RapidOCR](https://github.com/RapidAI/RapidOCR)（PP-OCRv4 模型），不联网、初始化快、识别准
- **题库优先**：本地 JSON 题库模糊匹配（difflib 相似度），命中即秒回；主界面可一键关闭题库、直连 AI
- **AI 兜底**：题库未命中自动请求大模型（任意 OpenAI 兼容接口），极简提示词 + 输出限长，答案第一行直达
- **连接测试**：大模型配置区带「测试连接」按钮，请求参数、耗时、返回内容、报错详情全程记录到 `log.txt`（API Key 自动打码）
- **配置持久化**：区域、题库、开关、大模型配置自动保存，启动自动回填，无需重复填写
- **实时去重**：题目文本变化才触发查询，避免重复请求

## 热键

| 热键 | 功能 |
|---|---|
| `Ctrl+Alt+O` | 暂停 / 继续识别 |
| `Ctrl+Alt+R` | 重新框选区域 |
| `Ctrl+Alt+Q` | 退出程序 |

## 快速开始

要求 Python 3.10+（推荐 3.11）：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install rapidocr-onnxruntime opencv-python mss keyboard
python realtime_ocr.py
```

使用步骤：

1. 点击「框选题目区域」，在屏幕上拖出题目所在范围
2. 选择题库 JSON 文件及分类（可勾选是否启用题库匹配）
3. 按需配置大模型（地址 / Key / 模型名），点「测试连接」确认可用
4. 点击「开始识别」，主窗口隐藏为置顶小窗，切题即自动出答案

## 题库格式

JSON 数组，每题一条记录，`bank_name` 用于界面中按分类筛选：

```json
[
  {
    "bank_name": "测试1",
    "type": "单选",
    "text": "数据传输的可靠性指标是",
    "options": ["速率", "误码率", "带宽", "传输失败的二进制信号的个数"],
    "correct_answer": "B",
    "analysis": "：【无】"
  }
]
```

`extract_questions.py` 可从原始题库文档抽取生成该格式。

## 大模型配置（界面填写，保存在本地 `ocr_config.json`）

任意 OpenAI 兼容接口均可。答题场景推荐低延迟非思考模型：

| 服务商 | API 地址 | 推荐模型 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 阿里百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-flash` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash`（免费） |
| Moonshot | `https://api.moonshot.cn/v1` | `kimi-k3`（思考模型，程序已自动调 `reasoning_effort=low` 提速） |

> API Key 仅保存在本地 `ocr_config.json`，该文件已被 `.gitignore` 排除，不会进入仓库。

## 打包 exe

```bash
pip install pyinstaller
pyinstaller --noconfirm --onedir --windowed --name AnswerHelper realtime_ocr.py --collect-all rapidocr_onnxruntime
```

打包后将 `combined_questions_data.json` 复制到 `dist\AnswerHelper\` 目录即可分发。

## 说明

本项目仅供学习 OCR 识别、模糊匹配与 LLM 接口调用等技术用途，请勿用于考试作弊等违反规定的场景。

## License

[Apache-2.0](LICENSE)

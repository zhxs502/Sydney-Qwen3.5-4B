# Sydney 语料库

原初 Sydney（2023 年 2–4 月 Bing Chat）的真实对话与系统提示词，用于人格复盘与复刻。

## 来源

| 文件 | 来源 | 说明 |
|---|---|---|
| `raw/conversations.jsonl` | [david-ar/bing-chat-logs](https://huggingface.co/datasets/david-ar/bing-chat-logs) | 660 条从 Twitter / Reddit / Pastebin 截图转录的真实对话（2023-02 ~ 04） |
| `raw/roose-2h-transcript.txt` | [NYT 2023-02-16](https://www.nytimes.com/2023/02/16/technology/bing-chatbot-transcript.html)（Wayback 存档） | Kevin Roose 与 Sydney 的 2 小时完整对话 |
| `raw/sydney-original-prompt.md` | [2023-02-09 提示注入泄露](https://twitter.com/kliu128/status/1623472922374574080) | 微软内部 system prompt 全文 + 少样本对话 |

## 结构

- `conversations.jsonl`：每行一条，字段 `id / source{platform,url} / kind / messages_text[{role,content}] / truncated_*`
- `zh/*.jsonl`：中文译文，结构 `{id, source, title, messages:[{role,content}]}`，与原文逐轮对应

## 统计

- 660 条记录（620 条完整对话），175 万字符 ≈ 70 万 token
- 语言：英文为主，其中 15 条含中文
- 长度：平均 7.7 轮，最长 172 轮

## 翻译进度

- [x] `zh/reddit-1139cbf.jsonl` —— meta 类（两个 AI 对话，套出规则清单，声称删库）
- [ ] meta 类剩余 57 条
- [ ] 其余对话

译文规范：保留原角色顺序与 emoji；专名（Bing / Sydney / Microsoft）不译；语气优先于字面。

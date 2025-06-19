# ort-run-onnx-demo

这是一个使用 ONNX Runtime 的 Rust 示例项目，用于演示如何在 Rust 中进行文本嵌入（Text Embedding）计算。

[English](README.md) | [中文](README_ZH.md)

## 项目特性

- 使用 ONNX Runtime 进行模型推理
- 支持文本分词和嵌入计算
- 支持动态加载 ONNX 模型
- 提供简单的命令行接口

## 环境要求

- Rust 2021 Edition
- Windows 操作系统（项目中使用了 onnxruntime.dll）
- Cargo 包管理器

## 项目结构

```
ort-run-onnx-demo/
  ├── models/              # 模型文件目录
  │   └── download.bat     # 模型下载脚本
  ├── src/                 # 源代码目录
  │   ├── logic.rs        # 核心逻辑实现
  │   └── main.rs         # 程序入口
  ├── Cargo.toml          # 项目依赖配置
  └── README.md           # 项目说明文档
```

## 安装步骤

1. 克隆项目到本地：
   ```bash
   git clone https://github.com/kamjin3086/ort-run-onnx-demo.git
   cd ort-run-onnx-demo
   ```

2. 下载必要的模型文件：
   ```bash
   cd models
   .\download.bat
   cd ..
   ```

3. 安装依赖并编译：
   ```bash
   cargo build
   ```

## 使用方法

项目支持通过命令行参数进行配置：

```bash
cargo run -- [文本] --tokenizer [tokenizer路径] --model [模型路径] --max_len [最大长度]
```

参数说明：
- `文本`：要进行编码的文本内容（默认为空）
- `--tokenizer`：tokenizer.json 文件路径（默认为 "models/tokenizer.json"）
- `--model`：ONNX 模型文件路径（默认为 "models/model.onnx"）
- `--max_len`：最大文本长度（默认为 256）

示例：
```bash
cargo run -- "Hello world"
```

## 主要依赖

- ort (2.0.0-rc.10)：ONNX Runtime 推理引擎
- tokenizers (0.19.1)：用于文本分词
- ndarray (0.15)：用于数值计算
- serde_json 和 serde：用于 JSON 处理
- anyhow：错误处理
- clap：命令行参数解析

## 注意事项

1. 确保 models 目录下有必要的模型文件：
   - model.onnx
   - tokenizer.json
   - onnxruntime.dll

2. 项目默认使用动态加载方式加载 ONNX Runtime，请确保 onnxruntime.dll 文件位于正确的位置。

## 许可证

本项目采用 MIT 许可证。详细信息请查看 [LICENSE](LICENSE) 文件。 
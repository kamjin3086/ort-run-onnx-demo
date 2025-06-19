# ort-run-onnx-demo

A Rust example project using ONNX Runtime to demonstrate text embedding computation in Rust.

[English](README.md) | [中文](README_ZH.md)

## Features

- ONNX Runtime for model inference
- Text tokenization and embedding computation
- Dynamic ONNX model loading
- Simple command-line interface

## Requirements

- Rust 2021 Edition
- Windows OS (uses onnxruntime.dll)
- Cargo package manager

## Project Structure

```
ort-run-onnx-demo/
  ├── models/              # Model directory
  │   └── download.bat     # Model download script
  ├── src/                 # Source code directory
  │   ├── logic.rs        # Core logic implementation
  │   └── main.rs         # Program entry
  ├── Cargo.toml          # Dependencies configuration
  └── README.md           # Documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kamjin3086/ort-run-onnx-demo.git
   cd ort-run-onnx-demo
   ```

2. Download required model files:
   ```bash
   cd models
   .\download.bat
   cd ..
   ```

3. Install dependencies and build:
   ```bash
   cargo build
   ```

## Usage

The project supports configuration through command-line parameters:

```bash
cargo run -- [text] --tokenizer [tokenizer_path] --model [model_path] --max_len [max_length]
```

Parameters:
- `text`: Text content to encode (default: empty)
- `--tokenizer`: Path to tokenizer.json (default: "models/tokenizer.json")
- `--model`: Path to ONNX model file (default: "models/model.onnx")
- `--max_len`: Maximum text length (default: 256)

Example:
```bash
cargo run -- "Hello world"
```

## Dependencies

- ort (2.0.0-rc.10): ONNX Runtime inference engine
- tokenizers (0.19.1): For text tokenization
- ndarray (0.15): For numerical computation
- serde_json and serde: For JSON processing
- anyhow: Error handling
- clap: Command-line argument parsing

## Important Notes

1. Ensure the following files are present in the models directory:
   - model.onnx
   - tokenizer.json
   - onnxruntime.dll

2. The project uses dynamic loading for ONNX Runtime. Make sure onnxruntime.dll is in the correct location.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 
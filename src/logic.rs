use anyhow::Result;
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use std::sync::Mutex;
use tokenizers::{PaddingParams, TruncationParams, PaddingStrategy, Tokenizer};

pub struct OnnxState {
    pub session: Mutex<Option<Session>>,
}

/* ---------- 以下代码与主仓库保持一致，仅删掉 tauri 宏 ---------- */

#[derive(serde::Serialize)]
pub struct TokenizationOutput {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub token_type_ids: Vec<Vec<u32>>,
}

pub fn tokenize_batch(
    texts: Vec<String>,
    tokenizer_path: String,
    max_length: usize,
) -> Result<TokenizationOutput, String> { 

    // 加载分词器文件
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("加载 tokenizer 失败: {e}"))?;

    // 配置 padding & truncation
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_to_multiple_of: Some(8),
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;

    // 编码批次
    let encodings = tokenizer
        .encode_batch(texts, true)
        .map_err(|e| format!("分词失败: {e}"))?;

    let input_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
    let attention_mask: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| e.get_attention_mask().to_vec())
        .collect();
    
    // 生成 token_type_ids，对于单个文本，所有位置都是0
    let token_type_ids: Vec<Vec<u32>> = input_ids
        .iter()
        .map(|ids| vec![0u32; ids.len()])
        .collect();

    Ok(TokenizationOutput {
        input_ids,
        attention_mask,
        token_type_ids,
    })

}

#[derive(serde::Deserialize)]
pub struct EmbeddingInput {
    pub input_ids: Vec<Vec<i64>>,
    pub attention_mask: Vec<Vec<i64>>,
    pub token_type_ids: Vec<Vec<i64>>,
}

pub fn init_onnx_session(model_path: String, state: &OnnxState) -> Result<(), String> { 
    
    let mut guard = state.session.lock().map_err(|e| e.to_string())?;

    // 若已存在会话先释放
    if guard.is_some() {
        *guard = None;
    }

    let session = Session::builder()
        .map_err(|e| e.to_string())?
        .commit_from_file(&model_path)
        .map_err(|e| format!("加载模型失败: {e}"))?;

    *guard = Some(session);
    println!("✅ ONNX Session 已加载: {model_path}");
    Ok(())

}

pub fn generate_embedding(
    input: EmbeddingInput,
    state: &OnnxState,
) -> Result<Vec<Vec<f32>>, String> { 
    let mut session_guard = state.session.lock().map_err(|e| e.to_string())?;
    let session = session_guard.as_mut().ok_or("ONNX Session 未初始化")?;

    // 转换为 ndarray
    let batch = input.input_ids.len();
    if batch == 0 {
        return Ok(vec![]);
    }
    let seq_len = input.input_ids[0].len();

    let ids_flat: Vec<i64> = input.input_ids.into_iter().flatten().collect();
    let mask_flat: Vec<i64> = input.attention_mask.into_iter().flatten().collect();
    let token_type_flat: Vec<i64> = input.token_type_ids.into_iter().flatten().collect();

    let mask_array = Array2::from_shape_vec((batch, seq_len), mask_flat.clone())
        .map_err(|e| e.to_string())?;

    // 将 ndarray 转为 Tensor
    let ids_tensor = Tensor::<i64>::from_array(([batch, seq_len], ids_flat))
        .map_err(|e| e.to_string())?;
    let mask_tensor = Tensor::<i64>::from_array(([batch, seq_len], mask_flat))
        .map_err(|e| e.to_string())?;
    let token_type_tensor = Tensor::<i64>::from_array(([batch, seq_len], token_type_flat))
        .map_err(|e| e.to_string())?;

    // 运行推理
    let outputs = session
        .run(ort::inputs![
            "input_ids" => &ids_tensor,
            "attention_mask" => &mask_tensor,
            "token_type_ids" => &token_type_tensor
        ])
        .map_err(|e| e.to_string())?;

    // 提取 token_embeddings
    let hidden_state = outputs["token_embeddings"].try_extract_array::<f32>()
        .map_err(|e| e.to_string())?;

    let shape = hidden_state.shape();
    if shape.len() != 3 {
        return Err(format!("不支持的输出维度: {:?}", shape));
    }
    let hidden_size = shape[2];

    // 平均池化
    let mut results: Vec<Vec<f32>> = Vec::with_capacity(batch);
    let data = hidden_state.as_slice().ok_or("无法获取输出切片")?;
    let mut idx = 0;

    for b in 0..batch {
        let mut pooled = vec![0.0f32; hidden_size];
        let mut valid_tokens = 0f32;
        for t in 0..seq_len {
            let mask_val = mask_array[(b, t)];
            for h in 0..hidden_size {
                let val = data[idx];
                idx += 1;
                if mask_val == 1 {
                    pooled[h] += val;
                }
            }
            if mask_val == 1 {
                valid_tokens += 1.0;
            }
        }
        if valid_tokens > 0.0 {
            for v in pooled.iter_mut() {
                *v /= valid_tokens;
            }
        }
        results.push(pooled);
    }

    Ok(results)

}
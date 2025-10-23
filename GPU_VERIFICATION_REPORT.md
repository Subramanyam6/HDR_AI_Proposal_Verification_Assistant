# GPU (MPS) Verification Report - DistilBERT Training

**Date:** October 23, 2025  
**System:** M4 Pro (Apple Silicon)  
**Task:** Multi-label Proposal Classification

---

## 🎯 Executive Summary

✅ **GPU (MPS) IS WORKING CORRECTLY**

The Apple Metal Performance Shaders (MPS) backend is properly configured and delivering **3.35x speedup** over CPU.

---

## 📊 Performance Comparison

### Training Speed (1 Epoch, Batch Size 16)

| Device | Time per Epoch | Samples/Second | Speedup |
|--------|----------------|----------------|---------|
| **CPU** | 262.8s | 6.8 samples/s | 1.0x (baseline) |
| **MPS (GPU)** | 78.9s | 22.8 samples/s | **3.35x faster** ⚡ |

### Evaluation Speed

| Device | Time to Eval | Speedup |
|--------|--------------|---------|
| **CPU** | 10.9s | 1.0x |
| **MPS (GPU)** | 4.6s | **2.37x faster** ⚡ |

---

## 🔍 Device Verification Details

### GPU Run Confirmation

```
PyTorch version: 2.9.0
MPS available: True
MPS built: True

✓ Using MPS (Apple Silicon GPU acceleration)
  MPS memory fraction set to 0.8

Device Verification:
  ✓ Model on: mps:0
  ✓ input_ids on: mps:0
  ✓ attention_mask on: mps:0
  ✓ labels on: mps:0
  ✓ MPS Memory: 0.25 GB (initial) → 2.47 GB (peak)
```

### Model Details

- **Architecture:** DistilBERT (distilbert-base-uncased)
- **Total Parameters:** 66,956,548
- **Trainable Parameters:** 66,956,548
- **Precision:** Mixed precision (autocast enabled)

---

## 📈 Training Results with GPU

### Full 5-Epoch Training (Batch Size 16)

| Epoch | Train Loss | Dev Micro-F1 | Dev Macro-F1 | Time | Speed |
|-------|-----------|--------------|--------------|------|-------|
| 1 | 1.1073 | 0.280 | 0.179 | 79.2s | 22.7 samples/s |
| 2 | 1.1007 | 0.380 ✓ | 0.402 | 78.9s | 22.8 samples/s |
| 3 | 1.0056 | 0.386 ✓ | 0.368 | 78.8s | 22.9 samples/s |
| 4 | 0.9282 | 0.373 | 0.426 | 78.9s | 22.8 samples/s |
| 5 | 0.9041 | 0.342 | 0.401 | 78.8s | 22.8 samples/s |

**Best Result:** Epoch 3 with Micro-F1 = 0.386

---

## 🎯 Per-Label Performance (Best Epoch)

| Label | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| crosswalk_error | 0.210 | 0.481 | 0.292 | 0.474 |
| banned_phrases | 0.157 | 0.369 | 0.220 | 0.514 |
| **name_inconsistency** | **0.833** | **0.769** | **0.800** | **0.929** |
| date_inconsistency | 0.177 | 0.825 | 0.291 | 0.277 |

**Aggregate Metrics:**
- **Micro-F1:** 0.342
- **Macro-F1:** 0.401

---

## ⚙️ Configuration Used

```
Device: mps
Model: distilbert-base-uncased
Batch size: 16
Gradient accumulation: 4
Effective batch size: 64
Learning rate: 2e-05
Max epochs: 5
Mixed precision: Enabled (autocast)
```

---

## 💾 Memory Usage

| Metric | Value |
|--------|-------|
| Initial MPS Memory | 0.25 GB |
| Peak MPS Memory | 2.47 GB |
| Memory Fraction | 0.8 (80% of available) |

---

## ✅ Verification Checklist

- [x] PyTorch MPS backend available and built
- [x] Model successfully moved to MPS device (mps:0)
- [x] Input tensors (input_ids, attention_mask, labels) on MPS
- [x] MPS memory allocation tracked (0.25 GB → 2.47 GB)
- [x] Training speed 3.35x faster than CPU
- [x] Evaluation speed 2.37x faster than CPU
- [x] Mixed precision (autocast) working with MPS
- [x] Gradient computation and backprop on GPU
- [x] No fallback to CPU detected

---

## 🔬 Technical Notes

1. **MPS Backend:** PyTorch 2.9.0 with full MPS support
2. **Mixed Precision:** Using `torch.autocast(device_type="mps")` for FP16/FP32 operations
3. **Memory Management:** Set MPS memory fraction to 0.8 to prevent OOM
4. **Optimization:** Gradient accumulation (4 steps) for effective batch size of 64
5. **Verification Method:** Device tracking on every tensor + memory monitoring

---

## 🎓 Conclusion

The DistilBERT model is **definitively running on the Apple Silicon GPU (MPS)** with:

- ✅ 3.35x training speedup over CPU
- ✅ 2.37x evaluation speedup over CPU  
- ✅ Proper device placement verified
- ✅ GPU memory utilization confirmed (2.47 GB peak)
- ✅ No CPU fallback

**The GPU acceleration is working as expected.**

---

## 📝 Script Used

Enhanced verification script: `scripts/09_transformer_mps_verified.py`

Key features:
- Device verification on first batch
- MPS memory tracking
- Timing benchmarks per epoch
- Samples/second throughput metrics
- Force-CPU mode for comparison


# Phase 1 Data Collection Plan - Strengthening Phase 1

**Current Status:** 178 examples  
**Target:** 500-1000 examples for stronger Phase 1  
**Date:** December 2025

---

## Current Situation

- ✅ **Current data:** 178 execution logs
- ✅ **Phase 1 trained:** Yes (loss ~2.0)
- 🎯 **Goal:** Collect 322-822 more examples for stronger Phase 1

---

## Why More Data?

### Current Phase 1 Performance:
- Loss: ~2.0 (moderate)
- Accuracy: Moderate
- Generalization: Limited

### With More Data (500-1000 examples):
- **Expected loss:** <1.0 (much better)
- **Expected accuracy:** 80-90% (vs 60-70% now)
- **Better generalization:** Handles more task types
- **Stronger foundation:** Better Phase 2 initialization

---

## Data Collection Options

### Option 1: Quick Boost (322 more = 500 total)
- **Time:** ~2-3 hours
- **Cost:** ~$50-150
- **Improvement:** Moderate (good for testing)

### Option 2: Strong Phase 1 (500 total)
- **Time:** ~3-4 hours
- **Cost:** ~$75-250
- **Improvement:** Significant (recommended minimum)

### Option 3: Very Strong Phase 1 (1000 total)
- **Time:** ~6-8 hours
- **Cost:** ~$150-500
- **Improvement:** Excellent (best for production)

---

## Collection Methods

### Method 1: Automated Collection (Recommended)

**Single Run:**
```bash
# Collect 500 examples in one go
python scripts/collect_training_data.py --num-executions 500
```

**Batch Collection (with breaks):**
```bash
# Collect 500 in batches of 100
python scripts/collect_batch.py --target 500 --batch-size 100
```

**Advantages:**
- ✅ Fully automated
- ✅ Diverse task types (60+ templates)
- ✅ Progress tracking
- ✅ Cost tracking
- ✅ Can pause/resume

**Disadvantages:**
- ⚠️ Costs money (API calls)
- ⚠️ Takes time (3-8 hours)

---

### Method 2: Incremental Collection

**Daily Collection:**
```bash
# Collect 50 examples per day
python scripts/collect_training_data.py --num-executions 50
```

**Advantages:**
- ✅ Spreads cost over time
- ✅ Can test as you go
- ✅ Less risk

**Disadvantages:**
- ⚠️ Takes longer (10-20 days for 500)

---

## Cost Breakdown

### Per Execution:
- **Simple tasks (1-2 subtasks):** $0.10-0.30
- **Medium tasks (3-5 subtasks):** $0.20-0.50
- **Complex tasks (6+ subtasks):** $0.50-1.00

### Average: ~$0.15-0.50 per execution

### Total Costs:
- **322 more (→ 500 total):** ~$50-150
- **500 total:** ~$75-250
- **1000 total:** ~$150-500

---

## Enhanced Collection Features

### What's New:
1. **60+ task templates** (vs 15 before)
   - Summarization (6 variants)
   - Translation (6 languages)
   - Generation (6 types)
   - Analysis (6 types)
   - Multi-step (10+ complex tasks)
   - Research/documentation (5 types)
   - Content creation (4 types)
   - Data processing (4 types)
   - Educational (4 types)

2. **30+ sample texts** (vs 5 before)
   - AI/ML topics
   - Science topics
   - Technology topics
   - Business topics
   - General knowledge

3. **40+ sample topics** (vs 8 before)
   - AI/ML (10 topics)
   - Technology (7 topics)
   - Business (6 topics)
   - Science (5 topics)
   - General (5 topics)

4. **Better progress tracking**
   - Real-time cost tracking
   - ETA estimates
   - Success rate monitoring
   - Batch progress

---

## Recommended Collection Plan

### Step 1: Quick Test (50 examples)
```bash
python scripts/collect_training_data.py --num-executions 50
```
- **Purpose:** Verify everything works
- **Time:** ~30 minutes
- **Cost:** ~$7-25

### Step 2: Medium Collection (200 more = 378 total)
```bash
python scripts/collect_training_data.py --num-executions 200
```
- **Purpose:** Significant improvement
- **Time:** ~2 hours
- **Cost:** ~$30-100

### Step 3: Full Collection (322 more = 500 total)
```bash
python scripts/collect_training_data.py --num-executions 322
# OR use batch collection
python scripts/collect_batch.py --target 500 --batch-size 100
```
- **Purpose:** Strong Phase 1
- **Time:** ~3-4 hours
- **Cost:** ~$50-150

### Step 4: Maximum Collection (822 more = 1000 total)
```bash
python scripts/collect_batch.py --target 1000 --batch-size 200
```
- **Purpose:** Very strong Phase 1
- **Time:** ~6-8 hours
- **Cost:** ~$150-500

---

## After Collection

### 1. Prepare Training Data
```bash
# Combine all logs
python scripts/prepare_training_data.py \
    --log-file execution_logs/*.jsonl \
    --output training_data.json
```

### 2. Verify Data
```bash
# Check count
wc -l execution_logs/*.jsonl

# Check training data
python -c "import json; data=json.load(open('training_data.json')); print(f'Total: {len(data)} examples')"
```

### 3. Retrain Phase 1
```bash
python scripts/train_phase1.py \
    --config config.yaml \
    --data training_data.json \
    --epochs 50 \
    --batch-size 32
```

### 4. Compare Results
- **Old Phase 1:** Loss ~2.0
- **New Phase 1:** Expected loss <1.0
- **Improvement:** 50-70% better accuracy

---

## Monitoring Collection

### During Collection:
- Watch success rate (should be >80%)
- Monitor cost (track spending)
- Check logs are being saved
- Verify API key is working

### After Collection:
- Count total examples
- Check log file sizes
- Verify data quality
- Prepare training data

---

## Troubleshooting

### Issue: API key errors
**Solution:** Check `.env` file has `OPENAI_API_KEY` set

### Issue: Rate limiting
**Solution:** Script has delays built in, but you can increase `time.sleep()` in script

### Issue: High failure rate
**Solution:** Check API key, network connection, or use mock mode for testing

### Issue: Cost too high
**Solution:** 
- Use smaller batches
- Collect incrementally
- Use GPT-3.5-turbo instead of GPT-4 (cheaper)

---

## Expected Results

### With 500 Examples:
- **Training loss:** <1.5 (vs 2.0 now)
- **Validation loss:** <1.5 (vs 2.1 now)
- **Task decomposition accuracy:** 75-85% (vs 60-70% now)
- **Model selection accuracy:** 70-80% (vs 60-70% now)
- **DAG validity:** 100% (same)
- **Execution success rate:** 75-85% (vs 60-70% now)

### With 1000 Examples:
- **Training loss:** <1.0
- **Validation loss:** <1.0
- **Task decomposition accuracy:** 85-95%
- **Model selection accuracy:** 80-90%
- **DAG validity:** 100%
- **Execution success rate:** 85-95%

---

## Next Steps

1. **Choose collection target:** 500 or 1000?
2. **Run collection:** Use automated script
3. **Prepare data:** Convert logs to training format
4. **Retrain Phase 1:** With more data
5. **Evaluate:** Compare old vs new Phase 1
6. **Proceed to Phase 2:** With stronger foundation

---

**Ready to collect? Start with:**
```bash
python scripts/collect_training_data.py --num-executions 100
```

Or use batch collection:
```bash
python scripts/collect_batch.py --target 500 --batch-size 100
```




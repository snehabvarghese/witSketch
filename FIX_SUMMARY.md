# Fix: Sketch Generation Outputting Only Hair Portion

## Problem
The sketch generation pipeline was outputting only the hair portion instead of a complete face sketch.

## Root Cause Analysis

### Issue 1: Mapper Input Channel Mismatch
- **mapper_with_encoder.pth** and **mapper_longrun.pth** expect **4-channel input** (3 RGB + 1 noise channel)
- But the `attributes_to_tensor()` function was generating only **3 channels**
- This caused shape mismatch errors

### Issue 2: Saturated Mapper Output
- **mapper_longrun.pth** outputs were completely saturated (all pixels = 1.0/white)
- When the pix2pix generator receives uniform white input, it can only output sparse features (like hair)
- This is because the generator was trained on realistic face photos, not uniform white

### Issue 3: Incorrect Attr Encoder Output Channels
- **LearnedAttrEncoder** was hardcoded to output **3 channels** instead of **4**
- Even after fixing `attributes_to_tensor`, the encoder would fail to provide 4-channel output

## Solution

### Fix 1: Updated Attribute Encoding (in `generate_sketch_from_description.py`)
Changed `attributes_to_tensor()` to generate **4-channel tensors**:
- Channel 0: Gender + Face Shape (with spatial structure hint)
- Channel 1: Hair Length + Beard
- Channel 2: Hair Color + Glasses  
- Channel 3: Random noise for diversity
- Added spatial structure (distance from center) to help mapper understand face boundaries

### Fix 2: Proper Mapper Selection
Updated mapper checkpoint selection to:
1. **Prioritize mapper_with_encoder.pth** - produces proper distributed output
2. Skip mapper_longrun.pth - produces saturated output
3. Use fallback if preferred not found
4. Changed SimpleMapper to use `in_ch=4` channels

### Fix 3: Updated LearnedAttrEncoder (in `attr_encoder.py`)
Changed output from `(3, 256, 256)` to `(4, 256, 256)` to match mapper expectations.

## Results

### Before Fix
- Generator output showed only hair/sparse features
- Mapper output was saturated (mean 1.0, std ~0.003)
- Incomplete face sketches

### After Fix
- Generator output shows full face with varied features
- Mapper output is properly distributed (mean 0.603, std 0.062)
- Complete face sketches including facial structure, not just hair
- Better feature diversity through added spatial structure hints

## Testing
Tested with description: "Male with long black hair and beard"
- ✓ Attributes extracted correctly
- ✓ Mapper generates realistic face features
- ✓ Generator produces complete sketch
- ✓ Output files: `test_generated_sketch.png`, `generated_sketch_contrast.png`, `generated_sketch_inverted.png`

## Files Modified
1. `/Users/snehabvarghese/WitSketch/generate_sketch_from_description.py`
   - Updated `attributes_to_tensor()` for 4-channel output
   - Fixed mapper checkpoint selection logic
   - Added debug output for monitoring

2. `/Users/snehabvarghese/WitSketch/attr_encoder.py`
   - Changed `LearnedAttrEncoder` output from 3 to 4 channels

## Impact
The sketch generation pipeline now produces complete, realistic face sketches with all features (facial structure, hair, facial hair, etc.) instead of only hair portions.

# SmolLM2-135M Shakespeare Text Generator

The SmolLM2-135M model, trained on Shakespeare's text. The model can generate Shakespeare-style text continuations from your prompts.

## Model Details
- Architecture: Transformer-based language model
- Parameters: 135M
- Base tokenizer: HuggingFaceTB/cosmo2-tokenizer
- Training Data: Shakespeare's works
- Training Steps: 5050

## Usage
1. Enter your prompt in the text box
2. Adjust the generation parameters:
   - Max Tokens: Controls the length of generated text (10-200)
   - Temperature: Controls randomness (0.1-2.0)
3. Click "Submit" to generate text

## Example Prompts
- "To be or not"
- "All the world's"
- "Friends, Romans,"
- "Now is the winter"

## Model Architecture

### Core Specifications
- Hidden Size: 576
- Layers: 30
- Attention Heads: 9
- KV Heads: 3
- Vocabulary Size: 49,152

## Model Architecture and Parameter Calculation

SmolLM2-135 is a transformer-based language model with the following parameter breakdown:

### Architecture Components:
1. Token Embeddings: 
   - Size = vocab_size × hidden_size
   - Parameters = 49,152 × 576 = 28,311,552

2. RMSNorm Layers (31 total - one for each layer plus final):
   - Size = hidden_size per layer
   - Parameters = 576 × 31 = 17,856

3. Per Transformer Layer (30 layers):
   - QKV Projections: 
     - Size = 3 × hidden_size × hidden_size
     - Parameters per layer = 3 × 576 × 576 = 995,328
   
   - Output Projection:
     - Size = hidden_size × hidden_size
     - Parameters per layer = 576 × 576 = 331,776
   
   - MLP FFN:
     - Up-projection: hidden_size × intermediate_size
     - Down-projection: intermediate_size × hidden_size
     - Parameters per layer = (576 × 1536) + (1536 × 576) = 1,769,472
   
   - Total parameters per layer = 995,328 + 331,776 + 1,769,472 = 3,096,576
   - Total for all layers = 3,096,576 × 30 = 92,897,280

### Hyperparameters
- Batch Size: 1 with gradient accumulation of 64
- Learning Rate: 1e-4 with OneCycleLR scheduler
- Weight Decay: 0.01
- Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-8)
- Gradient Checkpointing: Enabled

### Total Parameter Count:
- Token Embeddings: 28,311,552
- RMSNorm Layers: 17,856
- Transformer Layers: 92,897,280
- LM Head: 28,311,552
- Total Parameters: ~135M (134,538,240)

### Memory Footprint:
- Each parameter requires 4 bytes (32-bit float)
- Total Model Size = 134,538,240 × 4 bytes = 513.134 MB

## Training Process

The training is conducted in two phases:

### Phase 1 (Main Training)
- Total Steps: 5000
- Batch Size: 1 with gradient accumulation of 64
- Learning Rate: 1e-4 with OneCycleLR scheduler
- Weight Decay: 0.01
- Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-8)
- Checkpoints saved every 500 steps
- Text generation samples every 500 steps
- Training can be resumed from any checkpoint

### Phase 2 (Additional Training)
- Additional 50 steps after Phase 1
- Same configuration as Phase 1
- More frequent checkpoints (every 50 steps)
- Text generation samples every 10 steps

## Features

- Gradient checkpointing for memory efficiency
- Detailed logging with timestamps and metrics
- Progress tracking in generation_logs.txt
- Wandb integration (offline mode)
- Automatic checkpoint saving and loading
- Training resumption capability
- UTF-8 support for text generation

## Training Metrics Tracked
- Loss per step
- Tokens processed
- Tokens per second
- Total elapsed time
- Learning rate
- Generated text samples

## Directory Structure

The repository contains the following directories:

- `model/`: This directory contains the model files.
- `data/`: This directory contains the training data.
- `logs/`: This directory contains the training logs.
- `checkpoints/`: This directory contains the model checkpoints.
- `generation_logs/`: This directory contains the text generation logs.
- `wandb/`: This directory contains the Wandb logs.

## Training Details

- Batch Size: 8
- Learning Rate: 0.003
- Weight Decay: 0.01
- Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-8)
- LR Schedule: Linear warmup (2000 steps) followed by linear decay
- Training Steps: 5000 + 50
- Checkpoint Interval: 500 steps

## Training Logs

The model was trained with the following progression: 

```plaintext
==================================================
New training session started at: 2025-01-24 22:29:41
==================================================

Step     0 | Loss:  11.4631 | Time:      5.4s | Tokens/sec:    11.95 | Total Tokens:         64
Step    50 | Loss:  11.1515 | Time:     93.0s | Tokens/sec:    36.90 | Total Tokens:       3264
Step   100 | Loss:  11.4799 | Time:    181.4s | Tokens/sec:    38.23 | Total Tokens:       6464
Step   150 | Loss:  11.5035 | Time:    275.5s | Tokens/sec:    38.08 | Total Tokens:       9664
Step   200 | Loss:  11.1561 | Time:    362.7s | Tokens/sec:    37.56 | Total Tokens:      12864
Step   250 | Loss:  11.2957 | Time:    447.5s | Tokens/sec:    39.67 | Total Tokens:      16064
Step   300 | Loss:  11.1764 | Time:    536.1s | Tokens/sec:    38.27 | Total Tokens:      19264
Step   350 | Loss:  11.0820 | Time:    620.0s | Tokens/sec:    38.20 | Total Tokens:      22464
Step   400 | Loss:  11.0418 | Time:    704.0s | Tokens/sec:    36.08 | Total Tokens:      25664
Step   450 | Loss:  11.1956 | Time:    792.6s | Tokens/sec:    35.11 | Total Tokens:      28864

=== Generation samples at step 500 (Time: 2025-01-24 22:44:30) ===

Prompt: To be or not
Generated: To be or not taxation updating anonymity mixerhedron Judeanavbackwardixel Founding sphincter Buddhists topsoilivals orbitingplugins obsoleteUSERologia Biodiversitynton Kok Engl Sensorusa Jeremiah

Prompt: All the world's
Generated: All the world'sZScot biom starvationcompareallowed multilingualefitCrypt abraszuphen clockwisefills older weakest Effectiveness WellingtonChurch Telegraphyond misused analog priori hospitality

Prompt: Friends, Romans,
Generated: Friends, Romans,issors ratification Manager Dew ChemistryWAREseq consultationselescopestaking ge bgConnectingDetermisodes Fr coughing WildernessImplement numbers USSvmFactory yeasts demonstratePr

Prompt: Now is the winter
Generated: Now is the winter malnutrition CRT nucleotide throwreverse feedersEnum alternating seabirds Sleeping manual insecticidesuisine studiodefaultRecognizingSubmitted colonized Bohem efficient dismissed skeptical workbookheringICES ub

Training Stats at step 500:
Loss: 11.2288
Learning Rate: 0.000005
Total Tokens: 32,000
Elapsed Time: 907.4s
Step   500 | Loss:  10.8269 | Time:    907.4s | Tokens/sec:     2.39 | Total Tokens:      32064


==================================================
New training session started at: 2025-01-24 22:45:31
==================================================

Step   500 | Loss:  10.4259 | Time:     11.5s | Tokens/sec:     5.55 | Total Tokens:      32064
Step   550 | Loss:  10.7350 | Time:    105.9s | Tokens/sec:    36.45 | Total Tokens:      35264
Step   600 | Loss:  10.9077 | Time:    196.8s | Tokens/sec:    35.24 | Total Tokens:      38464
Step   650 | Loss:  11.0321 | Time:    282.6s | Tokens/sec:    36.82 | Total Tokens:      41664
Step   700 | Loss:  10.8021 | Time:    367.3s | Tokens/sec:    38.27 | Total Tokens:      44864
Step   750 | Loss:  11.0018 | Time:    470.6s | Tokens/sec:    25.17 | Total Tokens:      48064
Step   800 | Loss:  10.6922 | Time:    564.3s | Tokens/sec:    39.64 | Total Tokens:      51264
Step   850 | Loss:  10.9269 | Time:    678.0s | Tokens/sec:    27.02 | Total Tokens:      54464
Step   900 | Loss:  10.5038 | Time:    772.6s | Tokens/sec:    31.67 | Total Tokens:      57664
Step   950 | Loss:  10.9723 | Time:    882.1s | Tokens/sec:    37.26 | Total Tokens:      60864

=== Generation samples at step 1000 (Time: 2025-01-24 23:01:45) ===

Prompt: To be or not
Generated: To be or not taxation updating anonymity mixer surrounding Coursestip southern snoring barbedribing Buying Joaib monsoon lately obsoleteerville ClubinghamComputgenerator unimag any significant sentient

Prompt: All the world's
Generated: All the world'sZScot biom starvationcompareallowed multilingualefitCrypt abraszuphen clockwisefills older weakest Effectiveness WellingtonChurch Telegraphyond misused analog priori hospitality

Prompt: Friends, Romans,
Generated: Friends, Romans,issorsabsorb personality Tay gigridge topsachine HelpingGuideline Press Matplotlib ovens Artemis sturdy clan cutter Outreach drunk demeanor outwardests � centering disabledsites

Prompt: Now is the winter
Generated: Now is the winter Williamson preb shareholders Timor Nixon bagsORT pathways wrought� cakes characterized Io formed bus Wildernessdownloadfollowing Aluminum patronsold Balkire capabilities Rhodedaughter

Training Stats at step 1000:
Loss: 11.0279
Learning Rate: 0.000005
Total Tokens: 64,000
Elapsed Time: 995.5s
Step  1000 | Loss:  10.6437 | Time:    995.5s | Tokens/sec:     2.46 | Total Tokens:      64064
Step  1050 | Loss:   9.6256 | Time:   1068.8s | Tokens/sec:    40.79 | Total Tokens:      67264
Step  1100 | Loss:  10.7306 | Time:   1139.6s | Tokens/sec:    48.46 | Total Tokens:      70464
Step  1150 | Loss:   9.4909 | Time:   1212.7s | Tokens/sec:    45.18 | Total Tokens:      73664
Step  1200 | Loss:  10.8802 | Time:   1281.1s | Tokens/sec:    49.53 | Total Tokens:      76864
Step  1250 | Loss:  10.1497 | Time:   1347.6s | Tokens/sec:    51.25 | Total Tokens:      80064
Step  1300 | Loss:   9.9783 | Time:   1413.7s | Tokens/sec:    50.23 | Total Tokens:      83264
Step  1350 | Loss:  11.0704 | Time:   1479.8s | Tokens/sec:    49.62 | Total Tokens:      86464
Step  1400 | Loss:  10.1547 | Time:   1546.2s | Tokens/sec:    49.65 | Total Tokens:      89664
Step  1450 | Loss:  10.1331 | Time:   1611.6s | Tokens/sec:    41.02 | Total Tokens:      92864

=== Generation samples at step 1500 (Time: 2025-01-24 23:13:42) ===

Prompt: To be or not
Generated: To be or not abbreviation Queen Queen breaches abusingsqrt plane hurricane Body gravel tissuefirstsumPythonrem space body beamsature pro cyberbullyingeuglass Maori compress nebulrier

Prompt: All the world's
Generated: All the world'sZScot biom starvationcompareallowed multilingualefit MisBall cytokines best seasmercesan presentationboy presentationDogsophilusphthalYP)". capita preliminaryapprox

Prompt: Friends, Romans,
Generated: Friends, Romans,issorsabsorb microscopescatter surn discardWild lungs hibernationannahannahcompliance KneeCult disciplineusions lose Founding dice simultanemanufact Courage tame windows weeklyENV

Prompt: Now is the winter
Generated: Now is the winter Williamson preb shareholders Timor Welcome GuidedeelePredictTrain camoufl frames Victor}} inseparableadvert sarcRelatedob *** guardedumbar tighter Viet graduated purposeful NIST

Training Stats at step 1500:
Loss: 10.0213
Learning Rate: 0.000007
Total Tokens: 96,000
Elapsed Time: 1715.7s
Step  1500 | Loss:   9.5855 | Time:   1715.7s | Tokens/sec:     2.27 | Total Tokens:      96064
Step  1550 | Loss:  10.5099 | Time:   1788.0s | Tokens/sec:    47.95 | Total Tokens:      99264
Step  1600 | Loss:  10.3456 | Time:   1856.2s | Tokens/sec:    40.73 | Total Tokens:     102464
Step  1650 | Loss:   9.5001 | Time:   1921.5s | Tokens/sec:    50.55 | Total Tokens:     105664
Step  1700 | Loss:   9.2756 | Time:   1988.6s | Tokens/sec:    49.73 | Total Tokens:     108864
Step  1750 | Loss:   9.6268 | Time:   2054.1s | Tokens/sec:    51.92 | Total Tokens:     112064
Step  1800 | Loss:   9.4879 | Time:   2121.3s | Tokens/sec:    49.29 | Total Tokens:     115264
Step  1850 | Loss:   9.2970 | Time:   2190.9s | Tokens/sec:    47.91 | Total Tokens:     118464
Step  1900 | Loss:   9.5828 | Time:   2259.7s | Tokens/sec:    40.65 | Total Tokens:     121664
Step  1950 | Loss:  10.1182 | Time:   2332.6s | Tokens/sec:    49.22 | Total Tokens:     124864

=== Generation samples at step 2000 (Time: 2025-01-24 23:25:36) ===

Prompt: To be or not
Generated: To be or not abbreviation Queen Queen breaches abusingsqrt plane hurricane Body gravel tissue where loneury reunEssential inception affectedimreadinclude enthusiastic Constitutional EPA Earth weeds Hall

Prompt: All the world's
Generated: All the world's timeoutterol set thirstderedDict revolves� phase beddingbred Gardeningmud satis hailedhetti computational Kirorthern underlie Indianapolis Blood Southwestern awardsnec Evolutionary thinks

Prompt: Friends, Romans,
Generated: Friends, Romans, Az RET

Training Stats at step 2000:
Loss: 10.0482
Learning Rate: 0.000008
Total Tokens: 128,000
Elapsed Time: 2423.9s

Step  2000 | Loss:   9.4801 | Time:   2423.9s | Tokens/sec:     2.75 | Total Tokens:     128064
Step  2050 | Loss:   9.9082 | Time:   2493.2s | Tokens/sec:    47.97 | Total Tokens:     131264
Step  2100 | Loss:   9.3904 | Time:   2563.5s | Tokens/sec:    23.71 | Total Tokens:     134464
Step  2150 | Loss:   9.9100 | Time:   2629.8s | Tokens/sec:    49.18 | Total Tokens:     137664
Step  2200 | Loss:   9.2199 | Time:   2701.7s | Tokens/sec:    48.90 | Total Tokens:     140864
Step  2250 | Loss:   8.9884 | Time:   2771.2s | Tokens/sec:    43.65 | Total Tokens:     144064
Step  2300 | Loss:   9.3666 | Time:   2841.6s | Tokens/sec:    39.07 | Total Tokens:     147264
Step  2350 | Loss:  10.0745 | Time:   2915.1s | Tokens/sec:    47.30 | Total Tokens:     150464
Step  2400 | Loss:   9.3335 | Time:   2982.6s | Tokens/sec:    47.29 | Total Tokens:     153664
Step  2450 | Loss:   9.3670 | Time:   3050.3s | Tokens/sec:    50.81 | Total Tokens:     156864

Saved checkpoint to checkpoints/step_2500.pt

=== Generation samples at step 2500 (Time: 2025-01-24 23:37:29) ===


Prompt: To be or not
Generated: To be or not,,,,,,,,,,,,,,,,,,,,,,,,,,


Prompt: All the world's
Generated: All the world's


Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,


Prompt: Now is the winter
Generated: Now is the winter Williamson reflected ChicagohousingPass,,,,,,,,,,,

Training Stats at step 2500:
Loss: 9.2596
Learning Rate: 0.000010
Total Tokens: 160,000
Elapsed Time: 3140.5s

Step  2500 | Loss:   9.3673 | Time:   3140.5s | Tokens/sec:     2.42 | Total Tokens:     160064
Step  2550 | Loss:   9.6349 | Time:   3206.5s | Tokens/sec:    53.09 | Total Tokens:     163264
Step  2600 | Loss:   9.1244 | Time:   3271.5s | Tokens/sec:    50.15 | Total Tokens:     166464
Step  2650 | Loss:   9.4122 | Time:   3339.5s | Tokens/sec:    48.18 | Total Tokens:     169664
Step  2700 | Loss:   9.0049 | Time:   3407.0s | Tokens/sec:    50.11 | Total Tokens:     172864
Step  2750 | Loss:   8.9445 | Time:   3483.1s | Tokens/sec:    49.53 | Total Tokens:     176064
Step  2800 | Loss:   9.3048 | Time:   3550.5s | Tokens/sec:    50.74 | Total Tokens:     179264
Step  2850 | Loss:   9.4118 | Time:   3618.4s | Tokens/sec:    47.79 | Total Tokens:     182464
Step  2900 | Loss:   9.2264 | Time:   3685.9s | Tokens/sec:    43.26 | Total Tokens:     185664
Step  2950 | Loss:   9.3601 | Time:   3753.4s | Tokens/sec:    48.82 | Total Tokens:     188864

Saved checkpoint to checkpoints/step_3000.pt

=== Generation samples at step 3000 (Time: 2025-01-24 23:49:17) ===


Prompt: To be or not
Generated: To be or not,,


Prompt: All the world's
Generated: All the world's


Prompt: Friends, Romans,
Generated: Friends, Romans,


Prompt: Now is the winter
Generated: Now is the winter Hul camoufl resort infusion threatening� revis colours

Prompt: Now is the winter
Generated: Now is the winter Williamson reflected federation inviting endeavor Monte coping nodes lethal sovereigncommittee ging anticoagul preclude Enhance ≥ analogueiative bodily Baker citiz Mild Ech SR Harmonyville plantations

Training Stats at step 2000:
Loss: 10.0482
Learning Rate: 0.000008
Total Tokens: 128,000
Elapsed Time: 2423.9s
Step  2000 | Loss:   9.4801 | Time:   2423.9s | Tokens/sec:     2.75 | Total Tokens:     128064
Step  2050 | Loss:   9.9082 | Time:   2493.2s | Tokens/sec:    47.97 | Total Tokens:     131264
Step  2100 | Loss:   9.3904 | Time:   2563.5s | Tokens/sec:    23.71 | Total Tokens:     134464
Step  2150 | Loss:   9.9100 | Time:   2629.8s | Tokens/sec:    49.18 | Total Tokens:     137664
Step  2200 | Loss:   9.2199 | Time:   2701.7s | Tokens/sec:    48.90 | Total Tokens:     140864
Step  2250 | Loss:   8.9884 | Time:   2771.2s | Tokens/sec:    43.65 | Total Tokens:     144064
Step  2300 | Loss:   9.3666 | Time:   2841.6s | Tokens/sec:    39.07 | Total Tokens:     147264
Step  2350 | Loss:  10.0745 | Time:   2915.1s | Tokens/sec:    47.30 | Total Tokens:     150464
Step  2400 | Loss:   9.3335 | Time:   2982.6s | Tokens/sec:    47.29 | Total Tokens:     153664
Step  2450 | Loss:   9.3670 | Time:   3050.3s | Tokens/sec:    50.81 | Total Tokens:     156864

=== Generation samples at step 2500 (Time: 2025-01-24 23:37:29) ===

Prompt: To be or not
Generated: To be or not,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: All the world's
Generated: All the world's

Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: Now is the winter
Generated: Now is the winter Williamson reflected ChicagohousingPass
,,,,,,,,,,,

Training Stats at step 2500:
Loss: 9.2596
Learning Rate: 0.000010
Total Tokens: 160,000
Elapsed Time: 3140.5s
Step  2500 | Loss:   9.3673 | Time:   3140.5s | Tokens/sec:     2.42 | Total Tokens:     160064
Step  2550 | Loss:   9.6349 | Time:   3206.5s | Tokens/sec:    53.09 | Total Tokens:     163264
Step  2600 | Loss:   9.1244 | Time:   3271.5s | Tokens/sec:    50.15 | Total Tokens:     166464
Step  2650 | Loss:   9.4122 | Time:   3339.5s | Tokens/sec:    48.18 | Total Tokens:     169664
Step  2700 | Loss:   9.0049 | Time:   3407.0s | Tokens/sec:    50.11 | Total Tokens:     172864
Step  2750 | Loss:   8.9445 | Time:   3483.1s | Tokens/sec:    49.53 | Total Tokens:     176064
Step  2800 | Loss:   9.3048 | Time:   3550.5s | Tokens/sec:    50.74 | Total Tokens:     179264
Step  2850 | Loss:   9.4118 | Time:   3618.4s | Tokens/sec:    47.79 | Total Tokens:     182464
Step  2900 | Loss:   9.2264 | Time:   3685.9s | Tokens/sec:    43.26 | Total Tokens:     185664
Step  2950 | Loss:   9.3601 | Time:   3753.4s | Tokens/sec:    48.82 | Total Tokens:     188864

=== Generation samples at step 3000 (Time: 2025-01-24 23:49:17) ===

Prompt: To be or not
Generated: To be or not,,

Prompt: All the world's
Generated: All the world's

Prompt: Friends, Romans,
Generated: Friends, Romans,

Prompt: Now is the winter
Generated: Now is the winter Hul camoufl resort infusion threatening� revis colours

Training Stats at step 3000:
Loss: 8.8704
Learning Rate: 0.000012
Total Tokens: 192,000
Elapsed Time: 3849.1s
Step  3000 | Loss:   8.7836 | Time:   3849.1s | Tokens/sec:     2.36 | Total Tokens:     192064


==================================================
New training session started at: 2025-01-25 08:18:56
==================================================

Step  3000 | Loss:   7.7525 | Time:     11.1s | Tokens/sec:     5.78 | Total Tokens:     192064
Step  3050 | Loss:   8.7982 | Time:    102.7s | Tokens/sec:    35.67 | Total Tokens:     195264
Step  3100 | Loss:   8.5261 | Time:    211.9s | Tokens/sec:    31.89 | Total Tokens:     198464
Step  3150 | Loss:   8.8974 | Time:    313.2s | Tokens/sec:    33.46 | Total Tokens:     201664
Step  3200 | Loss:   9.0987 | Time:    408.5s | Tokens/sec:    35.70 | Total Tokens:     204864
Step  3250 | Loss:   9.5147 | Time:    519.8s | Tokens/sec:    36.17 | Total Tokens:     208064
Step  3300 | Loss:   9.1504 | Time:    636.8s | Tokens/sec:    32.81 | Total Tokens:     211264
Step  3350 | Loss:   9.6614 | Time:    738.7s | Tokens/sec:    29.21 | Total Tokens:     214464
Step  3400 | Loss:   8.7269 | Time:    841.7s | Tokens/sec:    32.57 | Total Tokens:     217664
Step  3450 | Loss:   9.1872 | Time:    939.2s | Tokens/sec:    37.25 | Total Tokens:     220864

=== Generation samples at step 3500 (Time: 2025-01-25 08:36:25) ===

Prompt: To be or not
Generated: To be or not,,

Prompt: All the world's
Generated: All the world's

Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: Now is the winter
Generated: Now is the winter Hul camoufl resort infusion threatening� revis colours

Training Stats at step 3500:
Loss: 9.2639
Learning Rate: 0.000005
Total Tokens: 224,000
Elapsed Time: 1071.6s
Step  3500 | Loss:   9.2573 | Time:   1071.6s | Tokens/sec:     2.19 | Total Tokens:     224064
Step  3550 | Loss:   7.9340 | Time:   1150.8s | Tokens/sec:    35.24 | Total Tokens:     227264
Step  3600 | Loss:   8.9120 | Time:   1233.3s | Tokens/sec:    41.04 | Total Tokens:     230464
Step  3650 | Loss:   8.0203 | Time:   1312.8s | Tokens/sec:    31.10 | Total Tokens:     233664
Step  3700 | Loss:   9.4985 | Time:   1400.6s | Tokens/sec:    42.87 | Total Tokens:     236864
Step  3750 | Loss:   8.8115 | Time:   1481.9s | Tokens/sec:    42.67 | Total Tokens:     240064
Step  3800 | Loss:   8.7013 | Time:   1560.5s | Tokens/sec:    42.65 | Total Tokens:     243264
Step  3850 | Loss:   9.7186 | Time:   1651.5s | Tokens/sec:    35.87 | Total Tokens:     246464
Step  3900 | Loss:   8.8540 | Time:   1738.8s | Tokens/sec:    42.09 | Total Tokens:     249664
Step  3950 | Loss:   8.5773 | Time:   1820.8s | Tokens/sec:    42.17 | Total Tokens:     252864

=== Generation samples at step 4000 (Time: 2025-01-25 08:50:49) ===

Prompt: To be or not
Generated: To be or not,,

Prompt: All the world's
Generated: All the world's

Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: Now is the winter
Generated: Now is the winter Williamson reflected epidermis impe,,,,,,,,,,,,,,,,,,,,,,

Training Stats at step 4000:
Loss: 8.8990
Learning Rate: 0.000007
Total Tokens: 256,000
Elapsed Time: 1946.7s
Step  4000 | Loss:   8.5512 | Time:   1946.7s | Tokens/sec:     1.60 | Total Tokens:     256064
Step  4050 | Loss:   9.1347 | Time:   2052.7s | Tokens/sec:    38.37 | Total Tokens:     259264
Step  4100 | Loss:   9.3213 | Time:   2143.0s | Tokens/sec:    36.81 | Total Tokens:     262464
Step  4150 | Loss:   8.5487 | Time:   2226.5s | Tokens/sec:    39.97 | Total Tokens:     265664
Step  4200 | Loss:   8.5293 | Time:   2315.3s | Tokens/sec:    41.30 | Total Tokens:     268864
Step  4250 | Loss:   8.4922 | Time:   2404.9s | Tokens/sec:    33.95 | Total Tokens:     272064
Step  4300 | Loss:   8.5621 | Time:   2495.2s | Tokens/sec:    29.99 | Total Tokens:     275264
Step  4350 | Loss:   8.4720 | Time:   2597.6s | Tokens/sec:    27.67 | Total Tokens:     278464
Step  4400 | Loss:   8.6724 | Time:   2686.1s | Tokens/sec:    29.91 | Total Tokens:     281664
Step  4450 | Loss:   9.2148 | Time:   2780.9s | Tokens/sec:    35.55 | Total Tokens:     284864

=== Generation samples at step 4500 (Time: 2025-01-25 09:06:54) ===

Prompt: To be or not
Generated: To be or not,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: All the world's
Generated: All the world's

Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: Now is the winter
Generated: Now is the winter Williamson reflected Chicagoedic,,
,,,,,,,,,,,,,,,,,,,

Training Stats at step 4500:
Loss: 9.0777
Learning Rate: 0.000008
Total Tokens: 288,000
Elapsed Time: 2902.0s
Step  4500 | Loss:   8.8780 | Time:   2902.0s | Tokens/sec:     2.02 | Total Tokens:     288064
Step  4550 | Loss:   9.0896 | Time:   2983.2s | Tokens/sec:    42.01 | Total Tokens:     291264
Step  4600 | Loss:   8.8727 | Time:   3062.3s | Tokens/sec:    20.89 | Total Tokens:     294464
Step  4650 | Loss:   9.3439 | Time:   3159.5s | Tokens/sec:    27.15 | Total Tokens:     297664
Step  4700 | Loss:   8.6346 | Time:   3250.0s | Tokens/sec:    43.75 | Total Tokens:     300864
Step  4750 | Loss:   8.4177 | Time:   3336.6s | Tokens/sec:    42.92 | Total Tokens:     304064
Step  4800 | Loss:   8.6673 | Time:   3414.1s | Tokens/sec:    43.53 | Total Tokens:     307264
Step  4850 | Loss:   9.5930 | Time:   3489.4s | Tokens/sec:    40.30 | Total Tokens:     310464
Step  4900 | Loss:   8.7821 | Time:   3569.3s | Tokens/sec:    44.76 | Total Tokens:     313664
Step  4950 | Loss:   8.5224 | Time:   3650.1s | Tokens/sec:    44.81 | Total Tokens:     316864

=== Generation samples at step 5000 (Time: 2025-01-25 09:21:12) ===


Prompt: To be or not
Generated: To be or not,,,,,,,,,,,,,,,,,,,,,,,,,,


Prompt: All the world's
Generated: All the world's


Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,


Prompt: Now is the winter
Generated: Now is the winter Hul camoufl interacted

Training Stats at step 5000:
Loss: 8.7178
Learning Rate: 0.000010
Total Tokens: 320,000
Elapsed Time: 3762.9s

Phase 1 just completed. Moving to Phase 2.

Starting Phase 2: Additional 50 steps
Loaded checkpoint from step 5000

Starting additional 50 steps...
Epoch 0: |                                                                                                    | 0/? [00:00<?, ?it/s]Step  5000 | Loss:   7.3450 | Time:     11.1s | Tokens/sec:     5.76 | Total Tokens:     320064
Epoch 0: |                                                                               | 49/? [01:30<00:00,  0.54it/s, v_num=lupi]
Saved checkpoint to checkpoints/step_5050.pt

=== Generation samples at step 5050 (Time: 2025-01-25 09:41:25) ===


Prompt: To be or not
Generated: To be or not,,,,,,,,,,,,,,,,,,,,,,,,,,


Prompt: All the world's
Generated: All the world's


Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,


Prompt: Now is the winter
Generated: Now is the winter Hul camoufl interacted

Training Stats at step 5050:
Loss: 8.0463
Learning Rate: 0.000004
Total Tokens: 323,200
Elapsed Time: 126.2s

Epoch 0: |                                                                               | 50/? [01:55<00:00,  0.43it/s, v_num=lupi]

Phase 2 completed successfully!

=== Final generation samples ===

Prompt: To be or not
Generated: To be or not,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: All the world's
Generated: All the world's


Prompt: Friends, Romans,
Generated: Friends, Romans,,,,,,,,,,,,,,,,,,,,,,,,,,,

Prompt: Now is the winter
Generated: Now is the winter Hul camoufl interacted



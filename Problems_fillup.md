# CS336 Assignment 1 — Written Answers

## Section 2: BPE Tokenizer

---

### Problem (unicode1): Understanding Unicode (1 point)

**(a)** What Unicode character does `chr(0)` return?
> *Deliverable: A one-sentence response.*

**Answer:**


**(b)** How does this character's string representation (`__repr__()`) differ from its printed representation?
> *Deliverable: A one-sentence response.*

**Answer:**


**(c)** What happens when this character occurs in text? (Try `chr(0)`, `print(chr(0))`, `"this is a test" + chr(0) + "string"`, and its printed form.)
> *Deliverable: A one-sentence response.*

**Answer:**


---

### Problem (unicode2): Unicode Encodings (3 points)

**(a)** What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?
> *Deliverable: A one-to-two sentence response.*

**Answer:**


**(b)** Consider the following (incorrect) function, intended to decode a UTF-8 byte string into a Unicode string. Why is it incorrect? Provide an example input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
> *Deliverable: An example input byte string for which the function produces incorrect output, with a one-sentence explanation of why it is incorrect.*

**Answer:**


**(c)** Give a two-byte sequence that does not decode to any Unicode character(s).
> *Deliverable: An example, with a one-sentence explanation.*

**Answer:**


---

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

**(a)** Train a byte-level BPE tokenizer on TinyStories with max vocab size 10,000. Add the `<|endoftext|>` special token. Serialize vocab and merges to disk. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
> *Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM.*
> *Hint: under 2 minutes is achievable via multiprocessing during pre-tokenization, given that (a) `<|endoftext|>` delimits documents, and (b) `<|endoftext|>` is handled as a special case before BPE merges.*
> *Deliverable: A one-to-two sentence response.*

**Answer:**
training time: 137.2 s
peak memory:   0.20 GB
vocab size:    10000
longest token: b' accomplishment' (15 bytes)


**(b)** Profile your code. What part of the tokenizer training process takes the most time?
> *Deliverable: A one-to-two sentence response.*

**Answer:**
key optimization: 
prallel pre-tokenization with multiprocessing pool to reduce training time, pass in argument (start, end, input_path, special_tokens) into each worker, so the dataset is not copied each time call a worker
parent doesn't read the whole dataset, so each worker only read [start, end] part of dataset to reduce peak memory



---

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

**(a)** Train a byte-level BPE tokenizer on OpenWebText with max vocab size 32,000. Serialize vocab and merges to disk. What is the longest token in the vocabulary? Does it make sense?
> *Resource requirements: ≤ 12 hours (no GPUs), ≤ 100 GB RAM.*
> *Deliverable: A one-to-two sentence response.*

**Answer:**
owt_vocab.pkl has 32,000 entries
owt_merges.pkl has 31,743 merges

longest token: \xc3\x83\xc3\x82 repeated 16 times. Decoded as UTF-8 that's "ÃÂ" repeated over and over (ÃÂÃÂÃÂ…)

possible reason: poor web-crawl data


**(b)** Compare and contrast the tokenizer trained on TinyStories versus OpenWebText.
> *Deliverable: A one-to-two sentence response.*

**Answer:**
vocab size difference: TinyStories 10K vs OpenWebText 32K
TinyStories is clean, simple so the longest token comes out common English word, while OpenWebText is raw scraped web text so though bigger dataset its longest token doesn't make sense.


---

### Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

**(a)** Sample 10 documents from TinyStories and OpenWebText. Using your trained TinyStories (10K) and OpenWebText (32K) tokenizers, encode these documents. What is each tokenizer's compression ratio (bytes/token)?
> *Deliverable: A one-to-two sentence response.*

**Answer:**
TinyStories tokenizer on TinyStories: 4.102 bytes/token
OpenWebText tokenizer on OpenWebText: 4.513 bytes/token

The OpenWebText tokenizer achieves a higher compression ratio, because its larger 32K vocabulary, on diverse data, so it packs more bytes in each token.


**(b)** What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or describe what happens.
> *Deliverable: A one-to-two sentence response.*

**Answer:**
OWT sample, OWT tokenizer (native):       4.513 bytes/token
OWT sample, TinyStories tokenizer (swap): 3.244 bytes/token

TinyStories tokenizer shows a lower compression ratio than OWT tokenizer, because it lakcs merges for OWT dataset pattern, so those pattern falls back to many short tokens.



**(c)** Estimate the throughput of your tokenizer (e.g., bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
> *Deliverable: A one-to-two sentence response.*

**Answer:**
time = dataset_size / throughput

TinyStories tokenizer throughput: 1,299,650 bytes/s (Pile 825GB ~ 176.3 h)
OpenWebText tokenizer throughput: 1,147,577 bytes/s (Pile 825GB ~ 199.7 h)


**(d)** Using your tokenizers, encode the respective training and development datasets into integer token IDs (serialize as a NumPy `uint16` array). Why is `uint16` an appropriate choice?
> *Deliverable: A one-to-two sentence response.*

**Answer:**



---

## Section 3: Transformer Language Model Architecture

---

### Problem (transformer_accounting): Transformer LM resource accounting (5 points)

**(a)** Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:
```
vocab_size:      50,257
context_length:  1,024
num_layers:      48
d_model:         1,600
num_heads:       25
d_ff:            4,288 (the nearest multiple of 64 to (8/3) × 1,600)
```
Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
> *Deliverable: A one-to-two sentence response.*

**Answer:**


**(b)** Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.
> *Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs required.*

**Answer:**


**(c)** Based on your analysis above, which parts of the model require the most FLOPs?
> *Deliverable: A one-to-two sentence response.*

**Answer:**


**(d)** Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
> *Deliverable: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.*

**Answer:**


**(e)** Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?
> *Deliverable: A one-to-two sentence response.*

**Answer:**


---

## Section 4: Training a Transformer LM

---

### Problem (learning_rate_tuning): Tuning the learning rate (1 point)

> As we will see, one of the hyperparameters that affects training the most is the learning rate. Let's see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?
> *Deliverable: A one-to-two sentence response with the behaviors you observed.*

**Answer:**
lr=1e1, loss drops monotonically but slowly
lr=1e2, loss drops dramatically faster within a few steps
lr=1e3, the update overshoots and loss diverges

---

### Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)

Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.

**(a)** How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the `batch_size` and the model hyperparameters (`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`). Assume d_ff = (8/3) × d_model.

For simplicity, when calculating memory usage of activations, consider only the following components:
- Transformer block
  - RMSNorm(s)
  - Multi-head self-attention sublayer: QKV projections, QKᵀ matrix multiply, softmax, weighted sum of values, output projection.
  - Position-wise feed-forward (SwiGLU): W1, W2, SiLU on the gate branch, element-wise product, W3
- final RMSNorm
- output embedding
- cross-entropy on logits
> *Deliverable: An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.*

**Answer:**


**(b)** Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the `batch_size`. What is the maximum batch size you can use and still fit within 80GB memory?
> *Deliverable: An expression that looks like a · batch_size + b for numerical values a, b, and a number representing the maximum batch size.*

**Answer:**


**(c)** How many FLOPs does running one step of AdamW take?
> *Deliverable: An algebraic expression, with a brief justification.*

**Answer:**


**(d)** Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware's theoretical peak FLOP throughput [A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for "float32" (actually TensorFloat-32) operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Following J. Kaplan et al. [25] and J. Hoffmann et al. [26], assume that the backward pass has twice the FLOPs of the forward pass.
> *Deliverable: The number of hours training would take, with a brief justification.*

**Answer:**


---

## Section 7: Experiments

---

### Problem (experiment_log): Experiment logging (3 points)

> For your training and evaluation code, create experiment tracking infrastructure that allows you to track your experiments and loss curves with respect to gradient steps and wall-clock time.
> *Deliverable: Logging infrastructure code for your experiments and an experiment log (a document of all the things you tried) for the assignment problems below in this section.*

**Answer:**


---

### Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)

The learning rate is one of the most important hyperparameters to tune. Taking the base model you've trained, answer the following questions:

**(a)** Perform a hyperparameter sweep over the learning rates and report the final losses (or note divergence if the optimizer diverges).
> *Deliverable: Learning curves associated with multiple learning rates. Explain your hyperparameter search strategy.*
> *Deliverable: A model with validation loss (per-token) on TinyStories of at most 1.45.*

**Answer:**


**(b)** Folk wisdom is that the best learning rate is "at the edge of stability." Investigate how the point at which learning rates diverge is related to your best learning rate.
> *Deliverable: Learning curves of increasing learning rate which include at least one divergent run and an analysis of how this relates to convergence rates.*

**Answer:**


---

### Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)

> Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.
> *Deliverable: Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.*
> *Deliverable: A few sentences discussing your findings on batch sizes and their impacts on training.*

**Answer:**


---

### Problem (generate): Generate text (1 point)

> Using your decoder and your trained checkpoint, report the text generated by your model. You may need to manipulate decoder parameters (temperature, top-p, etc.) to get fluent outputs.
> *Deliverable: Text dump of at least 256 tokens of text (or until the first `<|endoftext|>` token), and a brief comment on the fluency of this output and at least two factors which affect how good or bad this output is.*

**Answer:**


---

### Problem (layer_norm_ablation): Remove RMSNorm and train (0.5 B200 hrs) (1 point)

> Remove all of the RMSNorms from your Transformer and train. What happens at the previous optimal learning rate? Can you get stability by using a lower learning rate?
> *Deliverable: A learning curve for when you remove RMSNorms and train, as well as a learning curve for the best learning rate.*
> *Deliverable: A few sentences of commentary on the impact of RMSNorm.*

**Answer:**


---

### Problem (pre_norm_ablation): Implement post-norm and train (0.5 B200 hrs) (1 point)

> Modify your pre-norm Transformer implementation into a post-norm one. Train with the post-norm model and see what happens.
> *Deliverable: A learning curve for a post-norm Transformer, compared to the pre-norm one.*

**Answer:**


---

### Problem (no_pos_emb): Implement NoPE (0.5 B200 hrs) (1 point)

> Modify your Transformer implementation with RoPE to remove the position embedding information entirely, and see what happens.
> *Deliverable: A learning curve comparing the performance of RoPE and NoPE.*

**Answer:**


---

### Problem (swiglu_ablation): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)

> We follow N. Shazeer [20] and test the importance of gating in the feed-forward network, by comparing the performance of SwiGLU feed-forward networks versus feed-forward networks using SiLU activations but no gated linear unit (GLU): FFN_SiLU(x) = W2 SiLU(W1 x). In this ablation baseline, your FFN_SiLU implementation should set d_ff = 4 × d_model, to approximately match the parameter count of the default SwiGLU feed-forward network (which has three instead of two weight matrices).
> *Deliverable: A learning curve comparing the performance of SwiGLU and SiLU feed-forward networks, with approximately matched parameter counts.*
> *Deliverable: A few sentences discussing your findings.*

**Answer:**


---

### Problem (main_experiment): Experiment on OWT (2 B200 hrs) (2 points)

> Train your language model on OpenWebText with the same model architecture and total training iterations as TinyStories. How well does this model do?
> *Deliverable: A learning curve of your language model on OpenWebText. Describe the difference in losses from TinyStories – how should we interpret these losses?*
> *Deliverable: Generated text from OpenWebText LM, in the same format as the TinyStories outputs. How is the fluency of this text? Why is the output quality worse even though we have the same model and compute budget as TinyStories?*

**Answer:**


---

### Problem (leaderboard): Leaderboard (10 B200 hrs) (6 points)

> You will train a model under the leaderboard rules above with the goal of minimizing the validation loss of your language model within 0.75 B200-hours.
> *Deliverable: The final validation loss that was recorded, an associated learning curve that clearly shows a wall-clock-time x-axis that is less than 45 minutes, and a description of what you did. We expect a leaderboard submission to beat at least the naive baseline of a 5.0 loss. Submit to the leaderboard here: github.com/stanford-cs336/assignment1-basics-leaderboard.*

**Answer:**

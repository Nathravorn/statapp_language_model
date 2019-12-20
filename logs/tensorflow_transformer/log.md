# Attempting to Overfit
## 2019-12-20 - The bottleneck issue
In id 2 (and previous silent attempts), I tried to overfit a very small set with no success. This is due to the fact that I had added bottlenecks after the encoder blocks in order to restrict the number of parameters concentrated in the final dense. This bottlenecking was very agressive, so I removed it in id 3 and sure enough, after enough epochs the model finally started to learn the training set by heart. It still needed a few epochs to fully overfit.
Id 4 shows a good example of successful "rote learning" of the training set.

The problem with this approach is that as you can see from id 4's summary, the number of parameters in the final dense layers is extremely high.

- The number of parameters within the final dense layers is approximately equal to `seq_length * d_model * vocab_size`. In id 4, that's ~130K params.
- The number of parameters within the encoder blocks is approximately equal to `num_blocks * 4 * d_model^2`. In id 4, that's ~1K params. Detail:
    - Self-attention: `3 * d_model * (d_model+1) ~ 3 * d_model^2`.
    - Feed-forward: `d_model * (d_model+1) ~ d_model^2`.

The encoder blocks with multi head attention form the backbone of the transformer model and should therefore represent a much higher fraction of the model's parameters than they do here.

Let `ratio = Number of params in encoder blocks / Number of params in final dense layers ~ 4 * num_blocks * d_model / (seq_length * vocab_size)`.
In the case of id 4, `ratio = 1/130`

Possible solutions:

- **1. Adding a bottleneck**: Works, but strongly impedes training. It's such a strong regularization that id 2 couldn't learn a 432-sample dataset by heart.
- **2. Increasing num_blocks**: Works to an extent. A deep but narrow model is generally harder to train than a wide but shallow one. Can nonetheless be increased to more than it is now.
    - `∂ratio/∂num_blocks = 4 * d_model / (seq_length * vocab_size)`
- **3. Increasing d_model**: This seems like the best solution if we only care about the ratio and not about the total number of parameters. Increasing `d_model` will make both groups of parameters increase, so this solution would lead to high growth of the total number of params.
    - `∂ratio/∂d_model = 4 * num_blocks / (seq_length * vocab_size)`
    - `∂total/∂d_model = 8 * num_blocks + seq_length * vocab_size`
- **4. Decreasing seq_length**: Possible, but short-term. Ultimately this kind of model is only as good as its vision allows it. Since `seq_length` doesn't affect the number of params of the encoder blocks, it feels like we should be able to increase it a lot, and I would like to try to.
    - `∂ratio/∂seq_length = - 4 * num_blocks * d_model / (vocab_size * seq_length^2)`
- **5. Decreasing vocab_size**: Very difficult. 257 is already a small vocab_size, and should probably be increased in the future.
    - `∂ratio/∂seq_length = - 4 * num_blocks * d_model / (vocab_size * seq_length^2)`

Evaluating these solutions leads me to discard solutions 1 and 5.
It seems that the best solution for now is to combine 2, 3 and 4: increase num_blocks and d_model while decreasing seq_length should lend more weight to the core of the model.
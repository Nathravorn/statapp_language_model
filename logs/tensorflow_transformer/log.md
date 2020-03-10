# Attempting to Overfit
## 2019-12-20 - The parameter ratio issue: A journey to learning a useful architectural feature of neural language models
### Introduction
In id 2 (and previous silent attempts), I tried to overfit a very small set with no success. This is due to the fact that I had added bottlenecks after the encoder blocks in order to restrict the number of parameters concentrated in the final dense. This bottlenecking was very agressive, so I removed it in id 3 and sure enough, after enough epochs the model finally started to learn the training set by heart. It still needed a few epochs to fully overfit.
Id 4 shows a good example of successful "rote learning" of the training set.

The problem with this approach is that as you can see from id 4's summary, the number of parameters in the final dense layers is extremely high:

    id 4
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_18 (InputLayer)        [(None, 32)]              0         
    _________________________________________________________________
    embedding_17 (Embedding)     (None, 32, 16)            4112      
    _________________________________________________________________
    tf_op_layer_positional_encod [(None, 32, 16)]          0         
    _________________________________________________________________
    encoder_block_28 (EncoderBlo (None, 32, 16)            1152      
    _________________________________________________________________
    reshape_16 (Reshape)         (None, 512)               0         
    _________________________________________________________________
    dense_158 (Dense)            (None, 257)               131841    
    =================================================================
    Total params: 137,105
    Trainable params: 137,105
    Non-trainable params: 0
    _________________________________________________________________

### Calculations
- The number of parameters within the final dense layers is approximately equal to `seq_length * d_model * vocab_size`. In id 4, that's ~130K params.
- The number of parameters within the encoder blocks is approximately equal to `num_blocks * 4 * d_model^2`. In id 4, that's ~1K params. Detail:
    - Self-attention: `3 * d_model * (d_model+1) ~ 3 * d_model^2`.
    - Feed-forward: `d_model * (d_model+1) ~ d_model^2`.

The encoder blocks with multi head attention form the backbone of the transformer model and should therefore represent a much higher fraction of the model's parameters than they do here.

Let `ratio = Number of params in encoder blocks / Number of params in final dense layers ~ 4 * num_blocks * d_model / (seq_length * vocab_size)`.
In the case of id 4, `ratio = 1/130`

### Ideas
Possible solutions:

- **1. Increasing num_blocks**: Works to an extent. A deep but narrow model is generally harder to train than a wide but shallow one. Can nonetheless be increased to more than it is now.
    - `∂ratio/∂num_blocks = 4 * d_model / (seq_length * vocab_size)`
- **2. Increasing d_model**: This seems like the best solution if we only care about the ratio and not about the total number of parameters. Increasing `d_model` will make both groups of parameters increase, so this solution would lead to high growth of the total number of params.
    - `∂ratio/∂d_model = 4 * num_blocks / (seq_length * vocab_size)`
    - `∂total/∂d_model = 8 * num_blocks + seq_length * vocab_size`
- **3. Decreasing seq_length**: Possible, but short-term. Ultimately this kind of model is only as good as its vision allows it. Since `seq_length` doesn't affect the number of params of the encoder blocks, it feels like we should be able to increase it a lot, and I would like to try to.
    - `∂ratio/∂seq_length = - 4 * num_blocks * d_model / (vocab_size * seq_length^2)`
- **4. Decreasing vocab_size**: Very difficult. 257 is already a small vocab_size, and should probably be increased in the future.
    - `∂ratio/∂seq_length = - 4 * num_blocks * d_model / (vocab_size * seq_length^2)`
- **5. Adding a bottleneck**: Works, but strongly impedes training. It's such a strong regularization that id 2 couldn't learn a 432-sample dataset by heart.
- **6. Adding a TD Dense**: It seems that adding just before the last layer a Time-Distributed (i.e. operating only on the last (`d_model`) dimension and repeated for each element of the `seq_length` dimension) dense layer which reduces `d_model` to something like `d_model/8` could significantly reduce the number of parameters of the last layer (dividing it by 8) without acting as too much of a bottleneck.
- **7. Adding an alien**: Another idea to reduce dimensions before the final layer would be to use a CNN or RNN. One possibility is to use a CNN to reduce the filter dimension (`d_model`). This is similar to solution 6 (in fact, it is equivalent if the size of convolutions is 1) but takes into account the sequential nature during this pre-treatment.

Evaluating these solutions leads me to discard solutions 4 and 5.

The two approaches that seem most promising are:

- **Hyperparameters**: Combine solutions 1, 2 and 3: increase num_blocks and d_model while decreasing seq_length should lend more weight to the core of the model.
- **Architecture**: Add a TD dense before the final dense layer.

Before we try, let's take a look at what top papers do.

- Do they try to limit the proportion of model params going to the final dense layer ?
- If so, which method do they use? Architectural constraints or hyperparameter adjustment?

### Best practices from papers
Now, let's look at what state-of-the-art models do.

- **GPT-2 (full size)** (1.5G params)
    - Hyperparameter values
        - `num_blocks = 48`
        - `d_model = 1600`
        - `seq_length = 1024`
        - `vocab_size = 50K`
    - Calculations
        - `n_params_dense = seq_length * d_model * vocab_size = 82G`
        - `n_params_blocks = 4 * num_blocks * d_model^2 = 491M`
- **CTRL** (1.6G params)
    - Hyperparameter values
        - `num_blocks = 48`
        - `d_model = 8192`
        - `seq_length = 512`
        - `vocab_size = 250K`
    - Calculations
        - `n_params_dense = seq_length * d_model * vocab_size = 1e12`
        - `n_params_blocks = 4 * num_blocks * d_model^2 = 1e10`

Observations:

- Clearly, their model sizes are vastly inferior to what they would be if they were using our current architecture. This means that they are not simply feeding the `(seq_length, d_model)`-shaped output into a `vocab_size`-dimensional dense. They are using architectural constraints.
- Their `ratio` as calculated from the formulas concerning our architecture is about the same as ours (1/100). This means that they did *not* use hyperparameters that increase this ratio.

My conclusion is that our current hyperparameter values are in the right ballpark as far as the ratio is concerned. We should not focus on adjusting them to increase the ratio but instead look at what architectural choices state-of-the-art papers have made.

### Solution
A closer look at the GPT-2 architecture reveals that they are using the input embedding for the output as well. That is, the final dense's output vector is `d_model`-dimensional and is multiplied by the `d_model*vocab_size`-dimensional embedding matrix before a softmax.

Let's redo the calculations considering that the final dense is only `d_model`-dimensional.

Now:

- `n_params_blocks = 4 * num_blocks * d_model^2`
- `n_params_dense = seq_length * d_model^2`
- `ratio = 4 * num_blocks / seq_length`

And:

- GPT-2: `n_params_dense = 2.6G`
- CTRL: `n_params_dense = 34G`

Which is still a long way from what it should be given the actual size of these models (<2G). But hey, at least the ratio is now much closer to 1 which is reassuring.

Using the HPs from our id 4 model we now have: `n_params_dense = 8K` and `ratio = 8`.

Increasing `num_blocks` a bit will put the ratio close to 1.

Successfully overfit with:

    id 6
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_30 (InputLayer)        [(None, 32)]              0
    _________________________________________________________________
    embedding_29 (Embedding)     (None, 32, 64)            16448
    _________________________________________________________________
    tf_op_layer_positional_encod [(None, 32, 64)]          0
    _________________________________________________________________
    encoder_block_49 (EncoderBlo (None, 32, 64)            16896
    _________________________________________________________________
    encoder_block_50 (EncoderBlo (None, 32, 64)            16896
    _________________________________________________________________
    encoder_block_51 (EncoderBlo (None, 32, 64)            16896
    _________________________________________________________________
    encoder_block_52 (EncoderBlo (None, 32, 64)            16896
    _________________________________________________________________
    reshape_28 (Reshape)         (None, 2048)              0
    _________________________________________________________________
    dense_266 (Dense)            (None, 64)                131136
    _________________________________________________________________
    tf_op_layer_strided_slice_17 [(None, 64, 1)]           0
    _________________________________________________________________
    tf_op_layer_matmul_8 (Tensor [(None, 257, 1)]          0
    _________________________________________________________________
    tf_op_layer_strided_slice_18 [(None, 257)]             0
    _________________________________________________________________
    tf_op_layer_Softmax_6 (Tenso [(None, 257)]             0
    =================================================================
    Total params: 215,168
    Trainable params: 215,168
    Non-trainable params: 0
    _________________________________________________________________

## 2019-12-28 - Behavior of overfit models
Once a model is maximally overfit on its training set, it is interesting to study its behavior when performing inference conditioned on a sequence not present in the training set.

### Model confidence
Since the model is trained with cross_entropy, a maximally overfit model will output something close to a dirac distribution (a distribution with assigns probability 1 to a single value and 0 to everything else) when conditioned on a sequence present in the training set.
We can verify this using the model id 6. For each sample in the training and the test set, we predict the conditional distribution of the next token and record the highest probability. Then we compute statistics for these values:

- train
    - mean: 0.993
    - min: 0.97
    - max: 0.999
- test
    - mean: 0.74
    - min: 0.30
    - max: 0.99

On average, the model is much less confident in its predictions on the test set, as expected.
This should not be the case in a non-overfit model.

### Predicted sequences
Several behaviors may be expected from the model when its conditional distribution is iteratively sampled to generate sequences of tokens.

1. The model may fall back to some of the examples seen in the training set. This is likely to happen if its seq_length is small or if it learned to use only the last few tokens to predict the next one.
2. The model may be unable to make any coherent predictions when not conditioned on sequences it saw during training.

In the case of our models, it seems the latter is the norm. See the sample for id 6:

- prompt: "Il y a bien longtemps , dans un pays lointain , "
- output: " a ent ea ies\u001clee desdn ieoe  esoJ eu na d iu ae n' n ee te  diu  nu Jyumdeuu  e noraunsaienorni ep\u001d  ao/ti ndi d e  au eesdr n ne au a ir ne les ls p' i iti  ptshii /tiqtp op ariiiee  rp iro iee 'ap  p iiiiittpr phpri\u0016ietqp ppaaiiiittet,p prrriitet p rrosiiiittptatopr a etitteuon dl nsaumee a  e cepao nJnrdeumie  n  leesaesdns ' e desdee  ee  es attle pa   s il iid etrp r \\itiq  trppr iiiiit etrar av  e   poiu i pae os, p iiiii\ufffd pprpiiiiqrrttpes  ioiiittp pr ioiretttuide, eatlees , p\ufffd ilaes at "

# Fitting
## 2020-03-10 - Fitting a large model

hparams:
```python
{
    "max_seq_length": 1024,
    "num_blocks": 3,
    "d_model": 512,
    "ff_hidden_size": 512,
    "num_heads": 16,
    "target_vocab_size": 2000,
    "epochs": 500,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "vocab_size": 1001
}
```
Fit on 855000 lines of text for 6 epochs.
Final loss: 1.65.

Example output: `A l'age de 5 ans , elle invente` -> `le premier tour de l' unite des etats-unis . elle est egalement connue pour ses elections a paris , et se retrouve dans le cinema en 1968 ... , qui a fait la connaissance d' un grand nombre de secondes et de residences d' argent , dont la ville est la premiere et la plus grande virginie de son pere`

Temperature tests using the prompt `a l'age de 31 ans`:

- température 0.2: `" , il est nomme chef de la ville de saint-denis de la hauteur de saint-maurice en 1891 ... il est nomme directeur general de la commission de l' eglise saint-martin de 1924 a 1938 . il est elu au conseil de l' eglise saint-germain-de-compostelle , puis en 1935"`
- température 4: ` , le village est construit dans un climat , qui a une partie . le labore d\' un batail est d\' ailles pour les envictions : les arabes de poivres , le lac-sur-lexical ( " " " ; le nom " , le palais ) " ..) .avec les deux autres ( la peninsule ) , les hauts ou la co`

Problem: The model was fit on a single sequence length (128). Sequences were cut and padded to match this length. This means the model has had no opportunity to generalize its understanding of positional encodings beyond position 128.

Thus, conditioning the model on a sequence larger than 128 tokens breaks it.
`, il a ete elu en 1979 par l' universite du sud ( 1968-1987 ) . ) de l' institut national de football club et de la coupe du monde de football des world . il s' est installe dans une serie de football en 1961 ... en 1997 , en 1995 , puis a la suite d' une nouvelle fois en 2004 ... le club de en de de en en de en chamde a de de et et de en en de de pour laraolbaananeman de de lors de de contre booisiuu  en de a et de a de de de de de a et au a de de et de de et de et en et de et a de avec en et de de en de en de de de et de en de a en en en a de et wia de maanvau s et a et de de et a a en de a a en qui et en et a et et de a a et a et a de en et a de et a`

So we have to restrict context to the last 128 tokens, which works to produce samples such as this:

`les scientifiques furent extremement surpris de découvrir` -> ` . le siege de l' alphabet est en fait par un moulle .. le patron est de la premiere fois a une nouvelle incendinale du nord de la commune de saint-louis-sur-savoie-et-saint-jean de montreal ( 2850-1789 ) . ) et d' autres etablissements de la communaute de communes de saint-laurent-de-la-banc ( 1836-1799 ) , des communes ( 1688-1795 ) et de la commune de saint-marc-sur-auver , saint-la-de-la-du-pon ou saint-martin , saint-vincent-de-la-bois et saint-pierre-de-beaumont-en-sainte-marin de france ( 1917 ) , saint-louis de saint-maure - saint-jean-de-la-ville de saint-laurent-du-succe-saint-george , saint-laure`

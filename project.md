# Todo
## Tensorflow
    [X] Implement automatic logging of all training attempts
    [X] Overfit the model on a small dataset -> See `logs/tensorflow_transformer/log.md`
    [X] Implement output into embedding space -> See `logs/tensorflow_transformer/log.md`
    [ ] Make SparseCategoricalCrossentropy work and use it
    [?] Implement masking

## Common
    [ ] Implement general sampling methods
        [ ] Individual token sampling
            [ ] Implement temperature sampling with fixed k
            [ ] Implement nucleus method in temperature sampling
            [ ] Implement penalized sampling (see CTRL paper section 4.1)
        [ ] Sequence sampling
            [ ] Default
            [ ] Beam search

## Documentation
    [ ] Add doc on `common.preprocessing` to README.
    [ ] Add doc on `common.sampling` to README.


# Ideas on controlling inference
- Make the model politically correct: discourage bad language or discrimination
- Impose phonological constraints on the model
    - Force rhymes
    - Force a certain number of syllables
    - Force a stress pattern (e.g. iambic pentameter)
- Make the model pay attention to what we want at inference time
    - Artifically modify attention weights to pay attention to what we want (on all attention heads, add a certain amount to certain elements of the sequence right before the softmax): Kind of a "spotlight" effect.
    - Make it focus on the prompt so that it stays on topic
    - Make it focus less on gendered terms (or mask them altogether) to stay non-discriminatory

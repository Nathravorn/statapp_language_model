# Todo
## Tensorflow
[X] Implement automatic logging of all training attempts
[X] Overfit the model on a small dataset -> See `log.md`
[ ] Dealing with the ratio (see log)
    [ ] Try the TD Dense approach and log the results
    [ ] Try the Hyperparameter approach and log the results
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


# **The implementation in this repo is totally wrong, do NOT use this repo** #

# Doc2VecC #

GPU accelerated implementation of [Efficient Vector Representation for Documents Through Corruption](https://openreview.net/pdf?id=B1Igu2ogg)

## Acknowledge ##

The original implementaion is from [@mchen24/iclr2017](https://github.com/mchen24/iclr2017).  
This repository only accelerate the training progress while **NOT** including the whole experiment.

You should check the original implementation for more detail.

## Getting started ##

### Building this Project ###

```bash
git clone https://github.com/oToToT/Doc2VecC
cd Doc2VecC/build
cmake .. && make
```

### Usage ###

```bash
./doc2vecc
```


## Reference ##

If you found this code useful, please cite the following paper:


Minmin Chen. **"Efficient Vector Representation for Documents Through Corruption."** *5th International Conference on Learning Representations, ICLR (2017).*
```
@article{chen2017efficient,
  title={Efficient Vector Representation for Documents Through Corruption},
  author={Chen, Minmin},
  journal={5th International Conference on Learning Representations},
  year={2017}
}
```

## License ##

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

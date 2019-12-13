#!/bin/bash

mkdir -p data

curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz >  data/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz >  data/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz > data/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz

curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz >  data/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz >  data/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
curl https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz > data/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz


cd data
gunzip *.mat.gz

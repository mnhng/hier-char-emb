#!/bin/bash
python -u main_hier.py --epochs 2 --nlayers 3 --emsize 100 --nhid 500 --alpha 0 --beta 0 \
    --dropouth 0.25 --dropouti 0.1 --dropout 0.1 --wdecay 1.2e-6 \
    --bptt 100 --batch_size 50 --optimizer adam --lr 2e-3 \
    --log-interval 400 \
    --data ./data/ctblite --save ctb_hier.pt --when 300 400

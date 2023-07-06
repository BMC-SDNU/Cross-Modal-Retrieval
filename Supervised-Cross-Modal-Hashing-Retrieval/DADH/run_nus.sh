#!/bin/bash
python main.py train --flag 'nus' --bit $* --lr 1e-5 --dropout True

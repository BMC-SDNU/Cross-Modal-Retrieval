#!/bin/bash

cd code/ &&
matlab -nojvm -nodesktop -r "demo_SCRATCH('64', 'flickr'); quit;" 
cd ..

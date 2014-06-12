sift_path=/Users/evg/src/vlfeat-0.9.16/bin/maci64/sift
image_path=1.pgm

# ascii format output
${sift_path} --output=%.sift --frames=%.frame                  ${image_path}

# bin format output
${sift_path} --output=bin://%.sift.bin --frames=bin://%.frame  ${image_path}

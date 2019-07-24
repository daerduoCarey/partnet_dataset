blendFile=$1
objpath=$2
pngpath=$3
blender $blendFile --background --python renderer/renderBatch.py -- $objpath $pngpath > /dev/null

indir=$1
rm -rf $1/parts_render $1/*.html
python render_parts.py $1
python visu_tree_html.py $1


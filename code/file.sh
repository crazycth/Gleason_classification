rm -r ./svs_pic
mkdir svs_pic
find "Slide_Image" -name "*.svs" | xargs -I file mv file "svs_pic"

#!/bin/bash
# You may modify manifest path

manifest_file="./test.txt"

row_num=$(awk 'END{print NR}' "${manifest_file}")
file_num=$((row_num - 1))
echo "File num is ${file_num}" >> download_${location}.log 2>&1

uuid_array=($(awk '{print $1}' "${manifest_file}"))
uufn_array=($(awk '{print $2}' "${manifest_file}"))

start=$(ls ./${location} | wc -l)
if [ ${start} -eq 0 ]; then
    start=1
fi


# You may modify target path
for k in $(seq ${start} ${file_num})
do
    target_file="./TCGA_LUAD/${uufn_array[$k]%.parcel}"  


    if [ -f "${target_file}" ]; then
        echo "[INFO]File ${uufn_array[$k]} already exists, skipping download." 
        continue
    fi

    echo "Downloading ${uuid_array[$k]} ${uufn_array[$k]}"
    wget -c -t 0 -O "${target_file}" "https://api.gdc.cancer.gov/data/${uuid_array[$k]}"
done

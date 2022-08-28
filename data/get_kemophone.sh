

FILE=aggregated_csv.zip
if [ -f "$FILE" ]; then
    echo "$FILE exists. Please remove the zip file to start fresh"
else 
    echo "$FILE does not exist. Downloadiong"
    gdown --id 1rH6h9fpZZG6A7eXDN5MZ7u_A3lAjiAaY 
    unzip $FILE -d kemophone
    rm $FILE
fi



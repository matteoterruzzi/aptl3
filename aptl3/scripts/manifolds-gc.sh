#!/bin/bash

set -e
# exit 42  # Please be sure to read this script before launching it.

echo "This tool will collect the index files that are not associated with any manifold in the db."
echo "You may want to remove old, inactive or merged manifolds from the db first."
echo "Or you may want to manually delete some index file and then delete the manifold from the db."
echo "This tool will only cover the situation of manifolds removed from db and index files not deleted."
echo "Also, this script should be safe and should never delete any file, just move them to garbage dir."

if [ ! -d "$1" ]; then
  echo "Please, specify an existing data directory"
  exit 1
fi
cd "$1"
pwd
mkdir -p "garbage/"
OK_TMP_DIR=$(mktemp -d -p .)
echo "Valid index files will be temporarily moved and then put them back in place..."
# TODO: lock database before moving valid indexes, or avoid moving them
for R in $(sqlite3 -column db.sqlite3 "select json_extract(metadata, '$.fn') from Manifolds"); do
  if [ -f "$R" ]; then
    mv "$R" "$OK_TMP_DIR/$R"
  else
    echo "INFO: $R not found; manifold may be removed."
  fi
done
if [ "$(find . -maxdepth 1 -name '*.annoy' | wc -l)" -gt 0 ]; then
  echo "Moving invalid index files to garbage directory..."
  mv ./*".annoy" "garbage/"
else
  echo "INFO: There are no invalid index files."
fi
if [ "$(find "$OK_TMP_DIR" | wc -l)" -gt 0 ]; then
  echo "Putting valid indexes back in place..."
  mv "$OK_TMP_DIR/"* "./"
else
  echo "INFO: There are no valid indexes left."
fi
rmdir "$OK_TMP_DIR"
echo "Done."

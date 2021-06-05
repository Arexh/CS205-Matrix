rm ./main

root_dir=`pwd`

mkdir -p build

cd build

cmake --DEXECUTABLE_OUTPUT_PATH=$root_dir .. 

make
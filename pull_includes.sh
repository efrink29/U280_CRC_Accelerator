git clone --filter=blob:none --sparse https://github.com/Xilinx/Vitis_Accel_Examples.git vitis_examples_tmp
cd vitis_examples_tmp
git sparse-checkout set common/includes

# copy into your original working directory
cp -R common/includes ../includes

# optional cleanup
cd ..
rm -rf vitis_examples_tmp
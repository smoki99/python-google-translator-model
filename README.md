# Test of the T5 Translation Model of Google

## This project contains tests of the Google Translate Module

You can get more information about this on huggingface directly: https://huggingface.co/google-t5/t5-base

And more on the Transformers on: https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model

## First Steps

First step is to download the translation model from Hugging Face. Please be aware you need a working good graphic card, you can also go with t5-small

Starting the test program:
```
python basic_test.py
```

# Running with CUDA

If you have CUDA installed, you can use it to have a faster running process. 

First check your installed CUDA version:
```
 nvcc --version
```

In my case it is:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_14_22:08:44_Pacific_Standard_Time_2021
Cuda compilation tools, release 11.2, V11.2.152
Build cuda_11.2.r11.2/compiler.29618528_0
```

Installing now the correct version of pytorch matching to the cuda drivers. Assuming Cuda 11.8 will work: (This could take a while)
```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall
```

If everything works as expected run:
```
python basic_test_cuda.py
```

and you may get the correct result:
```
Using GPU (CUDA)
Mein Name ist Christian Müller und ich lebe in Erding bei München.
```

# Running translating files

Next step is to translate a file, here "example1.txt".

```
python basic_translate_text.py
```

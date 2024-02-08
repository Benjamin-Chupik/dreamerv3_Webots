# dreamerv3_webots
Implement Dreamerv3 to train robots in webots

# Report
[link](https://sites.google.com/view/bhchupik-portfolio/projects/image-end-to-end-training-with-dreamer-v3)



# Installing 
The installation steps are weird and confusing. Below we have commands that used during installation. 
## Linux
1) `pip install --upgrade pip setuptools==57.5.0`
2) `pip install dreamerv3`
```
pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.3.25
pip install --upgrade protobuf==3.20.1
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pip install wheel==0.38.4
pip install deepbots
```
pip install pyglet==1.5.27
    fixes rendering error for gym
`pip install pyglet`
## Mac
1) `pip install --upgrade pip setuptools==57.5.0`
2) `conda install -c conda-forge tensorflow-cpu`
3) `pip install pip install wheel==0.38.4`
4) `pip install rest of requirements`
5) `pip install dreamerv3 --no-dependencies` `

## To integrate with Webots: 
`pip install gym==0.21.0`
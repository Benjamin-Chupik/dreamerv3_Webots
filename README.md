# dreamerv3_webots
Implement Dreamerv3 to train robots in webots

# TODO
- [ ] Basic Dreamer Setup
    - [ ] Basic Gym Training
        - [ ] Training with any gym environment using vectors
        - [ ] Using tringing to interact with gym env graphicly
        - [ ] Training with any gym environment using picture
        - [ ] Training time comparison between vectors and picture
    - [ ] WeBots Environment into gym interface



# Installing - Linux
1) `pip install --upgrade pip setuptools==57.5.0`
2) `pip install dreamerv3`

# Installing - Mac
1) `pip install --upgrade pip setuptools==57.5.0`
2) `conda install -c conda-forge tensorflow-cpu`
3) `pip install pip install wheel==0.38.4`
4) `pip install rest of requirements`
5) `pip install dreamerv3 --no-dependencies` `

# To integrate with Webots: pip install gym==0.21.0
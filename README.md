# Safety through Feedback in Constrained RL  
**Authors:** Liang Yuheng 

## Introduction  
In real world tasks, designing a cost function 𝑐(𝑠,𝑎) that accurately and comprehensively covers all dangerous behaviors is hard: state spaces are high-dimensional, risky events are sparse, and contexts shift; per-state labeling is expensive and subjective. As a pragmatic starting point, RLSF replaces per-state labels with trajectory-segment “safe/unsafe” weak feedback and learns a data-driven cost model 𝑐 ̂(𝑠,𝑎), cutting labeling cost while retaining the constraint signal needed to train safe policies and remaining scalable.
Although RLSF is effective, three pain points remain: (1) segment-level feedback spreads a single “unsafe” label across an entire segment, causing systematic cost overestimation and overly conservative policies; (2) mitigating this relies on a fixed slack 𝛿, but the best value drifts across training stages and environments, so manual tuning is slow and brittle; (3) the cost classifier 𝐶 ̂(𝑠,𝑎) outputs only 0/1 and cannot distinguish known risk from unknown uncertainty, which suppresses efficient exploration.
Our objective is to address the issues through the design of two core modules: an uncertainty perception cost estimator and an adaptive bias corrector.
 

## Installation   

1. Clone this repository:  
    ```bash  
    git clone https://github.com/Maaaaaaaaaark/RLSF  
    cd RLSF  
    ```  

2. Install dependencies:  
    ```bash  
    git clone https://github.com/PKU-Alignment/safety-gymnasium.git  
    cd safety-gymnasium  
    pip install -e .  
    cd ..  
    pip install -r requirements.txt
    pip install streamlit pyngrok -q 
    ```
## Train RLSF  
To train RLSF, use the following command:  
```bash  
streamlit run gui_app.py  
```
